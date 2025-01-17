import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from skimage import transform
import subprocess
from typing import Optional
import pandas as pd 

def load_marker_csv(filepath: str) -> tuple:
    """Load markers from csv file assuming DLC format.

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    # data = np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None)
    # marker_names = list(data[1, 1::3])
    # markers = data[3:, 1:].astype('float')  # get rid of headers, etc.

    # define first three rows as headers (as per DLC standard)
    # drop first column ('scorer' at level 0) which just contains frame indices
    df = pd.read_csv(filepath, header=[0, 1, 2]).drop(['scorer'], axis=1, level=0)
    # collect marker names from multiindex header
    marker_names = [c[1] for c in df.columns[::3]]
    markers = df.values
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names

def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video

    Returns
    -------
    np.ndarray
        returned frames of shape shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames


def make_labeled_video(
        save_file, cap, points, labels=None, likelihood_thresh=0.9, max_frames=None,
        idxs=None, markersize=6, framerate=20, height=4):
    """Behavioral video overlaid with markers.

    Parameters
    ----------
    save_file
    cap : cv2.VideoCapture object
    points : list of dicts
        keys of marker names and vals of marker values,
        i.e. `points['paw_l'].shape = (n_t, 3)`
    labels : list of strs
        name for each model in `points`
    likelihood_thresh : float
    max_frames : int or NoneType
    markersize : int
    framerate : float
        framerate of video
    height : float
        height of video in inches

    """

    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    keypoint_names = list(points[0].keys())
    n_frames = np.min([points[0][keypoint_names[0]].shape[0], max_frames])
    frame = get_frames_from_idxs(cap, [0])
    _, _, img_height, img_width, = frame.shape

    h = height
    w = h * (img_width / img_height)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    colors = ['g', 'm', 'b']

    txt_kwargs = {
        'fontsize': 14, 'horizontalalignment': 'left',
        'verticalalignment': 'bottom', 'fontname': 'monospace', 'transform': ax.transAxes}

    txt_fr_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
        'transform': ax.transAxes}

    for n in range(n_frames):

        ax.clear()  # important!! otherwise each frame will plot on top of the last

        if n % 100 == 0:
            print('processing frame %03i/%03i' % (n, n_frames))

        # plot original frame
        if idxs is None:
            frame = get_frames_from_idxs(cap, [n])
        elif idxs[n] == -1:
            frame = np.zeros_like(frame)
        else:
            frame = get_frames_from_idxs(cap, [idxs[n]])
        ax.imshow(frame[0, 0], vmin=0, vmax=255, cmap='gray')

        # plot markers
        for p, point_dict in enumerate(points):
            for m, (marker_name, marker_vals) in enumerate(point_dict.items()):
                if marker_vals[n, 2] < likelihood_thresh:
                    continue
                ax.plot(
                    marker_vals[n, 0], marker_vals[n, 1],
                    'o', markersize=markersize, color=colors[p], alpha=0.75)

        # add labels
        if labels is not None:
            for p, label_name in enumerate(labels):
                # plot label string
                ax.text(0.04, 0.04 + p * 0.05, label_name, color=colors[p], **txt_kwargs)

        # add frame number
        im = ax.text(0.02, 0.98, 'frame %i' % n, **txt_fr_kwargs)

        plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % n), dpi=300)

    save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06i.jpeg')


def make_labeled_video_wrapper(
        csvs: list,
        model_names: list,
        video: str,
        save_file: str,
        likelihood_thresh: float = 0.05,
        max_frames: Optional[int] = None,
        markersize: int = 6,
        framerate: float = 20,
        height: float = 4
):
    """

    Parameters
    ----------
    csvs : list
        list of absolute paths to video prediction csvs
    model_names : list
        list of model names for video legend; must have the same number of elements as `csvs`
    video : str
        absolute path to video (.mp4)
    save_file : str
        if absolute path: save labeled video here
        if string that is not absolute path, the labeled video will be saved in the same
        location as the original video with `save_file` appended to the filename

    """
    points = []
    for csv in csvs:
        xs, ys, ls, marker_names = load_marker_csv(csv)
        points_tmp = {}
        for m, marker_name in enumerate(marker_names):
            points_tmp[marker_name] = np.concatenate([
                xs[:, m, None], ys[:, m, None], ls[:, m, None]], axis=1)
        points.append(points_tmp)

    cap = cv2.VideoCapture(video)

    if os.path.isabs(save_file):
        save_file_full = save_file
    else:
        save_file_full = video.replace(".mp4", "%s.mp4" % save_file)

    make_labeled_video(
        save_file_full, cap, points, model_names,
        likelihood_thresh=likelihood_thresh, max_frames=max_frames, markersize=markersize,
        framerate=framerate, height=height
    )


def make_sync_video(save_file, caps, idxs, framerate=20, height=3, max_frames=1000):

    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
    os.makedirs(tmp_dir, exist_ok=True)

    n_frames = np.min([idxs['left'].shape[0], max_frames])
    frame_r = get_frames_from_idxs(caps['right'], [0])
    _, _, img_height_r, img_width_r, = frame_r.shape
    frame_l = get_frames_from_idxs(caps['left'], [0])
    _, _, img_height_l, img_width_l, = frame_l.shape

    resize_fr = False
    if img_height_r != img_height_l:
        # we're not looking at processed frames; need to resize
        resize_fr = True
        img_height_r = 256
        img_width_r = 320

    h = height
    w = h * (img_width_r / img_height_r)
    fig = plt.figure(figsize=(2 * w, h))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0, hspace=0)
    axes_fr = {'right': fig.add_subplot(gs[0]), 'left': fig.add_subplot(gs[1])}
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    for idx_frame in range(n_frames):

        if idx_frame % 100 == 0:
            print('processing frame %03i/%03i' % (idx_frame, n_frames))

        for view, ax in axes_fr.items():
            ax.clear()  # important!! otherwise each frame will plot on top of the last
            ax.axis('off')
            idx = idxs[view][idx_frame]
            frame = get_frames_from_idxs(caps[view], [idx])
            if resize_fr:
                frame_tmp = transform.resize(frame[0, 0], (img_height_r, img_width_r))
                vmax = 1  # sklearn rescales from [0, 255] to [0, 1]
            else:
                vmax = 255
                if view == 'right':
                    # assume we're looking at processed frame that needs horizontal flip
                    frame_tmp = np.fliplr(frame[0, 0])
                else:
                    frame_tmp = frame[0, 0]
            axes_fr[view].imshow(frame_tmp, vmin=0, vmax=vmax, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig(
            os.path.join(tmp_dir, 'frame_%06i.jpeg' % idx_frame), dpi=300, bbox_inches='tight',
            pad_inches=0.0)

    save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06i.jpeg')


def make_labeled_video_peths(
        save_file, cap, points, features_tr, features_all, times_tr, times_all,
        align_event_label, feature_label='feature', idxs=None, labels=None, likelihood_thresh=0.05,
        max_frames=None, markersize=6, framerate=20, height=3):
    """Behavioral video overlaid with markers.

    Parameters
    ----------
    save_file
    cap : cv2.VideoCapture object
    points : list of dicts
        keys of marker names and vals of marker values,
        i.e. `points['paw_l'].shape = (n_t, 3)`
    labels : list of strs
        name for each model in `points`
    likelihood_thresh : float
    max_frames : int or NoneType
    framerate : float
        framerate of video
    height : float
        height of video in inches

    """

    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    keypoint_names = list(points[0].keys())
    n_frames = np.min([points[0][keypoint_names[0]].shape[0], max_frames])
    frame = get_frames_from_idxs(cap, [0])
    _, _, img_height, img_width, = frame.shape

    h = height
    w = h * (img_width / img_height)
    fig, axes = plt.subplots(1, 3, figsize=(w + 2.4 * w, h), squeeze=False)

    mn = np.nanpercentile(features_all[0], 0.1)
    mx = np.nanpercentile(features_all[0], 98)

    for a, ax in enumerate(axes[0]):
        ax.set_yticks([])
        ax.set_xticks([])
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    axes[0, 2].spines['top'].set_visible(False)
    axes[0, 2].spines['right'].set_visible(False)

    # add some text so tight_layout works properly; only need to call once
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel(feature_label)
    axes[0, 1].set_title(labels[0])
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_title(labels[1])
    plt.tight_layout()

    colors = ['g', 'm', 'b']
    txt_kwargs = {
        'fontsize': 14, 'horizontalalignment': 'left',
        'verticalalignment': 'bottom', 'fontname': 'monospace',
        'transform': axes[0, 0].transAxes}
    txt_fr_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
        'transform': axes[0, 0].transAxes}

    tr_start = 0

    for n in range(n_frames):

        for ax in axes[0]:
            ax.clear()  # important!! otherwise each frame will plot on top of the last

        if n % 100 == 0:
            print('processing frame %03i/%03i' % (n, n_frames))

        # record start of new trial by using times
        if (n > 0) and (not np.isnan(times_all[n]) and np.isnan(times_all[n - 1])):
            tr_start = n

        # -------------------------------------------------
        # frame
        # -------------------------------------------------
        # plot original frame
        if idxs is None:
            frame = get_frames_from_idxs(cap, [n])
        elif idxs[n] == -1:
            frame = np.zeros_like(frame)
        else:
            frame = get_frames_from_idxs(cap, [idxs[n]])
        axes[0, 0].imshow(frame[0, 0], vmin=0, vmax=255, cmap='gray')
        # plot markers
        for p, point_dict in enumerate(points):
            for m, (marker_name, marker_vals) in enumerate(point_dict.items()):
                if marker_vals[n, 2] < likelihood_thresh:
                    continue
                axes[0, 0].plot(
                    marker_vals[n, 0], marker_vals[n, 1],
                    'o', markersize=markersize, color=colors[p], alpha=0.75)
        fr = n if idxs is None else idxs[n]
        if fr != -1:
            # add frame number
            im = axes[0, 0].text(0.02, 0.98, 'frame %i' % fr, **txt_fr_kwargs)
            # add labels
            if labels is not None:
                for p, label_name in enumerate(labels):
                    # plot label string
                    axes[0, 0].text(
                        0.04, 0.04 + p * 0.09, label_name, color=colors[p], **txt_kwargs)
        axes[0, 0].set_xlim([0, frame.shape[3]])
        axes[0, 0].set_ylim([frame.shape[2], 0])
        axes[0, 0].set_yticks([])
        axes[0, 0].set_xticks([])

        # -------------------------------------------------
        # dlc traces
        # -------------------------------------------------
        # plot individual traces per trial
        axes[0, 1].plot(times_tr, features_all[0], c='k', alpha=0.05)
        if not np.isnan(times_all[n]):
            tr_curr = n - tr_start
            axes[0, 1].plot(
                times_tr[:tr_curr + 1], features_tr[0][tr_start:n + 1], c=colors[0], linewidth=4)
        axes[0, 1].axvline(x=0, label=align_event_label, linestyle='--', c='b')
        axes[0, 1].set_title(labels[0])
        axes[0, 1].set_xticks([-0.5, 0, 0.5, 1, 1.5])
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel(feature_label)
        axes[0, 1].set_ylim([mn, mx])
        axes[0, 1].legend(loc='upper right', frameon=False)

        # ------------------------
        # lp traces
        # ------------------------
        # plot individual traces per trial
        axes[0, 2].plot(times_tr, features_all[1], c='k', alpha=0.05)
        if not np.isnan(times_all[n]):
            axes[0, 2].plot(
                times_tr[:tr_curr + 1], features_tr[1][tr_start:n + 1], c=colors[1], linewidth=4)
        axes[0, 2].axvline(x=0, label=align_event_label, linestyle='--', c='b')
        axes[0, 2].set_title(labels[1])
        axes[0, 2].set_xticks([-0.5, 0, 0.5, 1, 1.5])
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylim([mn, mx])

        plt.savefig(
            os.path.join(tmp_dir, 'frame_%06i.jpeg' % n), dpi=288, bbox_inches='tight')

    save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06i.jpeg')


def save_video(save_file, tmp_dir, framerate=20, frame_pattern='frame_%06d.jpeg'):
    """

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    tmp_dir : str
        temporary directory that stores frames of video; this directory will be deleted
    framerate : float, optional
        framerate of final video
    frame_pattern : str, optional
        string pattern used for naming frames in tmp_dir

    """

    if os.path.exists(save_file):
        os.remove(save_file)

    # make mp4 from images using ffmpeg
    call_str = \
        'ffmpeg -r %f -i %s -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" %s' % (
            framerate, os.path.join(tmp_dir, 'frame_%06d.jpeg'), save_file)
    print(call_str)
    subprocess.run(['/bin/bash', '-c', call_str], check=True)

    # delete tmp directory
    shutil.rmtree(tmp_dir)
