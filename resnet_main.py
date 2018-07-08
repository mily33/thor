import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim
from ReplayBuffer import ReplayBuffer


Action = ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft']


def processing(image):
    image = cv2.resize(image, (299, 299))
    image = np.array([image], dtype=np.float32)
    image = image / 255
    return image


def resnet_fm(input_ph):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(
            input_ph, num_classes=None,
            is_training=False, reuse=tf.AUTO_REUSE)
        feature_map = tf.squeeze(net, axis=[1, 2])
    return feature_map


def model(state_ph, target_ph, ac_dim, scope):
    state_fm = resnet_fm(state_ph)
    target_fm = resnet_fm(target_ph)

    with tf.variable_scope(scope, reuse=False):
        current_fc = slim.fully_connected(state_fm, num_outputs=512, activation_fn=None, scope='current_fc')
        target_fc = slim.fully_connected(target_fm, num_outputs=512, activation_fn=None, scope='target_fc')

        feature_map = tf.concat([current_fc, target_fc], 1)

        embedding_fusion = slim.stack(feature_map, slim.fully_connected,
                                      [512, 512], scope='embedding_fusion')
        # output
        value = slim.fully_connected(embedding_fusion, 1, activation_fn=None, scope='value')
        logit = slim.fully_connected(embedding_fusion, ac_dim, activation_fn=None, scope='policy')

    return value, logit


def main():
    # ==================
    # Def Hyperparameter
    # ==================
    learning_rate = 0.001
    alpha = 0.5            # loss penalty
    ac_dim = 4
    gamma = 0.9            # discount factor
    tf.set_random_seed(1)

    # ===========
    # Build Model
    # ===========
    current_state = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    next_state = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    target = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    reward = tf.placeholder(shape=[None], dtype=tf.float32)
    done_ph = tf.placeholder(shape=[None], dtype=tf.float32)

    current_value, logit = model(current_state, target, ac_dim=ac_dim, scope='current_state')
    next_value, _ = model(next_state, target, ac_dim=ac_dim, scope='next_state')

    # =========== Update Actor ===========
    sy_ac_na = tf.placeholder(shape=[None], name='ac', dtype='int32')
    sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype='float32')

    sy_sample_na = tf.reshape(tf.multinomial(logit, 1), [-1])
    sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=logit)

    loss_actor = -tf.reduce_mean(sy_logprob_n * sy_adv_n)

    # =========== Update Critic ===========
    q_var = current_value
    target_q_var = reward + (1 - done_ph) * next_value * gamma

    loss_critic = tf.reduce_mean(tf.squared_difference(target_q_var, q_var))
    loss = alpha * loss_actor + (1 - alpha) * loss_critic
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current_state')
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)

    # =========== Update Target Network =========
    q_func_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current_state')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='next_state')
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # ==============
    #  Init Session
    # ==============

    model_path = '/home/mily/PycharmProjects/models/resnet_v2/resnet_v2_50.ckpt'
    init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables(), ignore_missing_vars=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    # ===========
    #   Run Env
    # ===========

    # ====== Initialize ======
    max_len_step = 1000
    max_iteration = 10
    min_timesteps_per_batch = 100
    iteration = 0
    # max_size = 10000
    # frame_num = 4
    done = 0.0
    reward_to_go = True
    update_frequence = 20
    # model_initialized = False
    # trajectory_len = 0
    # batch_size = 5000
    # replay_buffer = ReplayBuffer(size=max_size, frame_history_len=frame_num)

    # ======= Sample and Store Trajectories ========
    import ai2thor.controller
    controller = ai2thor.controller.Controller()
    controller.start()
    controller.step(dict(action='Initialize', gridSize=0.1))
    event = controller.reset('FloorPlan5')
    target_img = event.cv2image()
    target_img = processing(target_img)

    def init_state():
        return 'FloorPlan' + str(np.random.randint(1, 30))

    total_timesteps = 0
    while iteration < max_iteration:
        print('************** Iteration %i ***************' % iteration)
        timesteps_this_batch = 0
        paths = []
        while True:
            obs = controller.reset(init_state())
            last_obs = obs.cv2image()
            ob = processing(last_obs)
            obs, acs, rewards, next_obs, dones = [], [], [], [], []
            steps = 0
            while True:
                obs.append(ob[0])
                ac = sess.run(sy_sample_na, feed_dict={current_state: ob, target: target_img})
                ac = ac[0]
                acs.append(ac)
                event = controller.step(dict(action=Action[ac]))
                next_ob = processing(event.cv2image())
                next_obs.append(next_ob[0])
                if (ob == target_img).all():
                    rew = 10.0
                    done = 1.0
                else:
                    rew = -0.1

                rewards.append(rew)
                dones.append(done)
                steps += 1
                if done or steps > max_len_step:
                    break
            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(acs),
                    "next_ob": np.array(next_obs),
                    "done": np.array(dones)}
            paths.append(path)
            timesteps_this_batch += len(path['reward'])
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        iteration += 1

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        next_ob_no = np.concatenate([path['next_ob'] for path in paths])
        rew_no = np.concatenate([path['reward'] for path in paths])
        done_mask = np.concatenate([path['done'] for path in paths])
        target_ob_no = np.concatenate([target_img for i in range(ob_no.shape[0])])

        q_n = []
        for path in paths:
            r = path["reward"]
            max_step = len(r)
            if reward_to_go:
                q = [np.sum(np.power(gamma, np.arange(max_step - t)) * r[t:]) for t in range(max_step)]
            else:
                q = [np.sum(np.power(gamma, np.arange(max_step)) * r) for _ in range(max_step)]
            q_n.extend(q)
        q_n = (q_n - np.mean(q_n, axis=0)) / (np.std(q_n, axis=0) + 1e-7)
        # while True:
        #     index = replay_buffer.store_frame(last_obs)
        #     q_input = replay_buffer.encode_recent_observation()
        #
        #     if np.random.random() < 0.9 or not model_initialized:
        #         idx = np.random.randint(0, 3)
        #         action = Action[idx]
        #     else:
        #         action_val = sess.run(logit, feed_dict={current_state: [q_input], target: target_img})[0]
        #         idx = np.argmax(action_val)
        #         action = Action[idx]
        #
        #     event = controller.step(dict(action=action))
        #     new_state = event.cv2image()
        #     new_state = processing(new_state)
        #     if last_obs == new_state:
        #         done = True
        #         reward = 10.0
        #     else:
        #         reward = -0.1
        #
        #     replay_buffer.store_effect(index, idx, reward, done)
        #     last_obs = new_state
        #     if trajectory_len > max_len_step:
        #         done = True
        #
        #     if done:
        #         controller.reset(init_state())
        #         trajectory_len = 0
        #         done = False
        #
        # # ========= Train the network ========
        # if replay_buffer.can_sample(batch_size=batch_size):
        #     obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size=batch_size)
        #     if not model_initialized:
        #         init_session()
        #         model_initialized = True

    #
    # writer = tf.summary.FileWriter('./log/')
    # writer.add_graph(sess.graph)
        l_actor = sess.run(loss_actor, feed_dict={current_state: ob_no,
                                                  target: target_ob_no,
                                                  sy_ac_na: ac_na,
                                                  sy_adv_n: q_n})
        l_critic = sess.run(loss_critic, feed_dict={current_state: ob_no,
                                                    target: target_ob_no,
                                                    next_state: next_ob_no,
                                                    done_ph: done_mask,
                                                    reward: rew_no})
        print(l_actor, l_critic)
        sess.run(update_op, feed_dict={current_state: ob_no,
                                       target: target_ob_no,
                                       sy_ac_na: ac_na,
                                       sy_adv_n: q_n,
                                       next_state: next_ob_no,
                                       done_ph: done_mask,
                                       reward: rew_no
                                       })
        if iteration % update_frequence == 0:
            sess.run(update_target_fn)


if __name__ == '__main__':
    main()
