import gradio as gr
import numpy as np

from rtmlib import Body, Wholebody, draw_skeleton

cached_model = {}


def predict(img,
            openpose_skeleton,
            model_type,
            black_bg=False,
            backend='onnxruntime',
            device='cpu'):

    if model_type == 'body':
        constructor = Body
    elif model_type == 'wholebody':
        constructor = Wholebody
    else:
        raise NotImplementedError

    model_id = str((constructor.__qualname__, openpose_skeleton, black_bg,
                    backend, device))

    if model_id in cached_model:
        model = cached_model[model_id]
    else:
        model = constructor(to_openpose=openpose_skeleton,
                            backend=backend,
                            device=device)
        cached_model[model_id] = model

    keypoints, scores = model(img)

    if black_bg:
        img_show = np.zeros_like(img, dtype=np.uint8)
    else:
        img_show = img.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.4)
    return img_show[:, :, ::-1]


with gr.Blocks() as demo:

    with gr.Tab('Upload-Image'):
        input_img = gr.Image(type='numpy')
        button = gr.Button('Inference', variant='primary')
        openpose_skeleton = gr.Checkbox(label='openpose-skeleton',
                                        info='Draw OpenPose-style Skeleton')
        black_bg = gr.Checkbox(
            label='black background',
            info='Whether to draw skeleton on black background')
        model_type = gr.Dropdown(['body', 'wholebody'],
                                 label='Keypoint Type',
                                 info='Body / Wholebody',
                                 value='body')
        backend = gr.Dropdown(['opencv', 'onnxruntime'],
                              label='Choose backend',
                              info='opencv / onnxruntime',
                              value='opencv')
        device = gr.Dropdown(['cpu', 'cuda'],
                             label='Choose device',
                             info='cpu / cuda',
                             value='cpu')

        gr.Markdown('## Output')
        out_image = gr.Image(type='numpy')
        gr.Examples(['./demo.jpg'], input_img)

        button.click(predict, [
            input_img, openpose_skeleton, model_type, black_bg, backend, device
        ], out_image)

gr.close_all()
demo.queue()
demo.launch()
