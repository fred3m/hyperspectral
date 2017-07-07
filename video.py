# paste this in an iPython notebook cell

from tempfile import NamedTemporaryFile
import base64

VIDEO_TAG = """<video controls>
            <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
            Your browser does not support the video tag.
            </video>"""

def anim_to_html(anim, fps):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode('utf-8')

    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim, fps=20):
    from IPython.display import HTML
    plt.close(anim._fig)
    return HTML(anim_to_html(anim, fps))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

feature = 3

Sx = hist["X", 1][:-1,feature]
vmin = np.min(Sx)
vmax = np.max(Sx)

im = plt.imshow(hist["X", 1][0][0].reshape(img_shape), animated=True, vmin=vmin, vmax=vmax)
tx = ax.set_title('Frame 0')
fig.colorbar(im)

def updatefig(idx):
    im.set_data(Sx[idx].reshape(img_shape))
    im.set_clim([vmin, vmax])
    tx.set_text('Frame {0}'.format(idx))
    return im,

anim = animation.FuncAnimation(fig, updatefig, interval=200, blit=True, frames=len(Sx), repeat_delay=1000)
print("Feature: {0}".format(list(points.keys())[feature]))
display_animation(anim, 5)