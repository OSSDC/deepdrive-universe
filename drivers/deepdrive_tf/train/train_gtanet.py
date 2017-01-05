import numpy as np
import tensorflow as tf
import os

from drivers.deepdrive_tf.train.data_utils import get_dataset
from drivers.deepdrive_tf.gtanet import GTANetModel

# settings
flags = tf.flags
flags.DEFINE_string("logdir", "/tmp/gtanet", "Logging directory.")
flags.DEFINE_string("data_path", os.environ('DEEPDRIVE_HDF5_PATH'), "Data path.")
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
flags.DEFINE_string("mode", "train", "Whether to run 'train' or 'eval' model.")
# flags.DEFINE_integer("num_gpus", 8, "Number of GPUs used.")
FLAGS = flags.FLAGS

def visualize_model(model, y):
    names = ["spin", "direction", "speed", "speed_change", "steering", "throttle"]
    for i in range(6):
        p = tf.reduce_mean(model.p[:, i])
        tf.scalar_summary("losses/{}/p".format(names[i]), tf.reduce_mean(p))
        err = 0.5 * tf.reduce_mean(tf.square(model.p[:, i] - y[:, i]))
        tf.scalar_summary("losses/{}/error".format(names[i]), err)
    tf.image_summary("model/x", model.x, max_images=10)

def visualize_gradients(grads_and_vars):
    grads = [g for g, v in grads_and_vars]
    var_list = [v for g, v in grads_and_vars]
    for g, v in grads_and_vars:
        if g is None:
            continue
        tf.histogram_summary(v.name, v)
        tf.histogram_summary(v.name + "/grad", g)
        tf.scalar_summary("norms/" + v.name, tf.global_norm([v]))
        tf.scalar_summary("norms/" + v.name + "/grad", tf.global_norm([g]))
    grad_norm = tf.global_norm(grads)
    tf.scalar_summary("model/grad_global_norm", grad_norm)
    tf.scalar_summary("model/var_global_norm", tf.global_norm(var_list))

def run():
    batch_size = 64
    num_targets = 6
    image_shape = (227, 227, 3)
    x = tf.placeholder(tf.float32, (None,) + image_shape)
    y = tf.placeholder(tf.float32, (None, num_targets))

    with tf.variable_scope("model") as vs:
        model = GTANetModel(x, num_targets)
        vs.reuse_variables()
        eval_model = GTANetModel(x, num_targets, is_training=False)

    # TODO: add polyak averaging.
    l2_norm = tf.global_norm(tf.trainable_variables())
    loss = 0.5 * tf.reduce_sum(tf.square(model.p - y)) / tf.to_float(tf.shape(x)[0])
    tf.scalar_summary("model/loss", loss)
    tf.scalar_summary("model/l2_norm", l2_norm)
    total_loss = loss + 0.0005 * l2_norm
    tf.scalar_summary("model/total_loss", total_loss)
    opt = tf.train.AdamOptimizer(2e-4)
    grads_and_vars = opt.compute_gradients(total_loss)
    visualize_model(model, y)
    visualize_gradients(grads_and_vars)
    summary_op = tf.merge_all_summaries()
    train_op = opt.apply_gradients(grads_and_vars, model.global_step)

    init_op = tf.initialize_all_variables()
    pretrained_var_map = {}
    for v in tf.trainable_variables():
        found = False
        for bad_layer in ["fc6", "fc7", "fc8"]:
            if bad_layer in v.name:
                found = True
        if found:
            continue

        pretrained_var_map[v.op.name[6:]] = v
        print(v.op.name, v.get_shape())
    alexnet_saver = tf.train.Saver(pretrained_var_map)

    def init_fn(ses):
        print("Initializing parameters.")
        ses.run(init_op)
        alexnet_saver.restore(ses, "bvlc_alexnet.ckpt")

    saver = tf.train.Saver()
    sv = tf.train.Supervisor(is_chief=True,
                             logdir=FLAGS.logdir + "/train",
                             summary_op=None,  # Automatic summaries don"t work with placeholders.
                             saver=saver,
                             global_step=model.global_step,
                             save_summaries_secs=30,
                             save_model_secs=60,
                             init_op=None,
                             init_fn=init_fn)

    eval_sw = tf.train.SummaryWriter(FLAGS.logdir + "/eval")
    train_dataset = get_dataset(FLAGS.data_path)
    eval_dataset = get_dataset(FLAGS.data_path, train=False)
    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config) as sess, sess.as_default():
        train_data_provider = train_dataset.iterate_forever(batch_size)
        while True:
            for i in range(1000):
                images, targets = next(train_data_provider)
                if i % 100 == 0 and i > 0:
                    _, summ = sess.run([train_op, summary_op], {x: images, y: targets})
                    sv.summary_computed(sess, summ)
                    sv.summary_writer.flush()
                else:
                    sess.run(train_op, {x: images, y: targets})

            step = model.global_step.eval()
            # Do evaluation
            losses = []
            for images, targets in eval_dataset.iterate_once(batch_size):
                preds = sess.run(eval_model.p, {x: images})
                losses += [np.square(targets - preds)]
            losses = np.concatenate(losses)
            print("Eval: shape: {}".format(losses.shape))
            summary = tf.Summary()
            summary.value.add(tag="eval/loss", simple_value=float(0.5 * losses.sum() / losses.shape[0]))
            names = ["spin", "direction", "speed", "speed_change", "steering", "throttle"]
            for i in range(len(names)):
                summary.value.add(tag="eval/{}".format(names[i]), simple_value=float(0.5 * losses[:, i].mean()))
            print(summary)
            eval_sw.add_summary(summary, step)
            eval_sw.flush()

if __name__ == "__main__":
    run()
