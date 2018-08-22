from PIL import Image
import visdom

vis = visdom.Visdom()

def reset(env):
    vis.close(env=env)

def visualize_samples(title, sample, env):
    vis.images(
        sample,
        opts=dict(title=title),
        env=env
        )
    vis.save([env])
