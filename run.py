import os
import base64
import gradio as gr
import json
import shutil
import random
import time
from PIL import Image
from io import BytesIO
from base import *
from params import *
from embeddings import *
from clip_config import *
from threading import Thread
import plotly.express as px
import numpy as np
from flask import Flask, send_file, request as flask_request

flask_app = Flask(__name__)

USER_SESSIONS_DIR = os.path.abspath("user_sessions")
os.makedirs(USER_SESSIONS_DIR, exist_ok=True)
print(f"User sessions directory: {USER_SESSIONS_DIR}")

default_examples = examples.copy()
default_images = images.copy()
default_coords = coords.copy()
user_data = {}

@flask_app.route('/plot/<session_hash>')
def serve_user_plot(session_hash):
    html_path = os.path.abspath(os.path.join("user_sessions", session_hash, "embedding_plot.html"))
    print(f"Trying to serve file at: {html_path}")
    print(f"File exists: {os.path.exists(html_path)}")
    
    if os.path.exists(html_path):
        try:
            return send_file(html_path, mimetype='text/html')
        except Exception as e:
            print(f"Error serving file: {e}")
            return f"Error serving file: {e}", 500
    else:
        return f"Plot not found at {html_path}", 404

def run_flask_server():
    try:
        print("Starting Flask server on port 8050")
        flask_app.run(host='0.0.0.0', port=8050, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error starting Flask server: {e}")

flask_thread = Thread(target=run_flask_server)
flask_thread.daemon = True
flask_thread.start()

def generate_user_html(session_hash):
    user_dir = os.path.abspath(os.path.join("user_sessions", session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    html_path = os.path.join(user_dir, "embedding_plot.html")
    
    user_fig = user_data[session_hash]["fig"]
    
    user_fig.write_html(
        html_path,
        full_html=True,
        include_plotlyjs='cdn',
        config={'responsive': True}
    )
    
    print(f"Generated HTML at: {html_path}")
    print(f"File exists after generation: {os.path.exists(html_path)}")
    
    try:
        os.chmod(html_path, 0o644)
    except Exception as e:
        print(f"Warning: Could not set file permissions: {e}")
    
    return html_path

def init_user_session(request: gr.Request):
    session_hash = request.session_hash
    if not session_hash:
        session_hash = str(random.randint(10000, 99999))
    
    print(f"Initializing session for: {session_hash}")
    
    if session_hash not in user_data:
        user_data[session_hash] = {
            "examples": default_examples.copy(),
            "images": default_images.copy(),
            "coords": default_coords.copy(),
            "axis": axis.copy(),
            "axis_names": axis_names.copy()
        }
        
        user_fig = px.scatter_3d(
            x=user_data[session_hash]["coords"][:, 0],
            y=user_data[session_hash]["coords"][:, 1],
            z=user_data[session_hash]["coords"][:, 2],
            labels={
                "x": user_data[session_hash]["axis_names"][0],
                "y": user_data[session_hash]["axis_names"][1],
                "z": user_data[session_hash]["axis_names"][2],
            },
            text=user_data[session_hash]["examples"],
            height=750,
        )
        
        user_fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0), 
            scene_camera=dict(eye=dict(x=2, y=2, z=0.1))
        )
        
        user_fig.update_traces(hoverinfo="none", hovertemplate=None)
        user_data[session_hash]["fig"] = user_fig
    
    html_path = generate_user_html(session_hash)
    
    timestamp = int(time.time())
    flask_url = f"http://localhost:8050/plot/{session_hash}?t={timestamp}"
    
    return flask_url, session_hash

def update_user_fig(session_hash):
    user_data[session_hash]["fig"].data[0].x = user_data[session_hash]["coords"][:, 0]
    user_data[session_hash]["fig"].data[0].y = user_data[session_hash]["coords"][:, 1]
    user_data[session_hash]["fig"].data[0].z = user_data[session_hash]["coords"][:, 2]
    user_data[session_hash]["fig"].data[0].text = user_data[session_hash]["examples"]
    
    user_data[session_hash]["fig"].update_layout(
        scene=dict(
            xaxis_title=user_data[session_hash]["axis_names"][0],
            yaxis_title=user_data[session_hash]["axis_names"][1],
            zaxis_title=user_data[session_hash]["axis_names"][2],
        )
    )
    
    html_path = generate_user_html(session_hash)
    
    timestamp = int(time.time())
    return f"http://localhost:8050/plot/{session_hash}?t={timestamp}"

def add_word_user(new_example, session_hash):
    user_examples = user_data[session_hash]["examples"]
    user_coords = user_data[session_hash]["coords"]
    user_images = user_data[session_hash]["images"]
    user_axis = user_data[session_hash]["axis"]
    
    new_coord = get_concat_embeddings([new_example]) @ user_axis.T
    new_coord[:, 1] = 5 * (1.0 - new_coord[:, 1])
    user_data[session_hash]["coords"] = np.vstack([user_coords, new_coord])
    
    image = pipe(
        prompt=new_example,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    user_data[session_hash]["images"].append("data:image/jpeg;base64, " + encoded_image)
    user_data[session_hash]["examples"].append(new_example)
    
    return update_user_fig(session_hash)

def remove_word_user(word_to_remove, session_hash):
    user_examples = user_data[session_hash]["examples"]
    user_coords = user_data[session_hash]["coords"]
    user_images = user_data[session_hash]["images"]
    
    examplesMap = {example: index for index, example in enumerate(user_examples)}
    if word_to_remove not in examplesMap:
        return update_user_fig(session_hash)
        
    index = examplesMap[word_to_remove]
    
    user_data[session_hash]["coords"] = np.delete(user_coords, index, 0)
    user_data[session_hash]["images"].pop(index)
    user_data[session_hash]["examples"].pop(index)
    
    return update_user_fig(session_hash)

def add_rem_word_user(new_examples, session_hash):
    new_examples = new_examples.replace(",", " ").split()
    
    for new_example in new_examples:
        if new_example in user_data[session_hash]["examples"]:
            remove_word_user(new_example, session_hash)
            gr.Info(f"Removed {new_example}")
        else:
            tokens = tokenizer.encode(new_example)
            if len(tokens) != 3:
                gr.Warning(f"{new_example} not found in embeddings")
            else:
                add_word_user(new_example, session_hash)
                gr.Info(f"Added {new_example}")
    
    return update_user_fig(session_hash)

def change_word_user(examples, session_hash):
    examples = examples.replace(",", " ").split()
    
    for example in examples:
        if example in user_data[session_hash]["examples"]:
            remove_word_user(example, session_hash)
            add_word_user(example, session_hash)
            gr.Info(f"Changed image for {example}")
    
    return update_user_fig(session_hash)

def clear_words_user(session_hash):
    if session_hash in user_data:
        while user_data[session_hash]["examples"]:
            remove_word_user(user_data[session_hash]["examples"][-1], session_hash)
        return update_user_fig(session_hash)
    return ""

def set_axis_user(axis_name, which_axis, from_words, to_words, session_hash):
    if axis_name != "residual":
        from_words, to_words = (
            from_words.replace(",", " ").split(),
            to_words.replace(",", " ").split(),
        )
        axis_emb = get_axis_embeddings(from_words, to_words)
        user_data[session_hash]["axis"][axisMap[which_axis]] = axis_emb
        user_data[session_hash]["axis_names"][axisMap[which_axis]] = axis_name
        
        for i, name in enumerate(user_data[session_hash]["axis_names"]):
            if name == "residual":
                user_data[session_hash]["axis"][i] = calculate_residual(
                    user_data[session_hash]["axis"], 
                    user_data[session_hash]["axis_names"], 
                    from_words, 
                    to_words, 
                    i
                )
                user_data[session_hash]["axis_names"][i] = "residual"
    else:
        residual = calculate_residual(
            user_data[session_hash]["axis"],
            user_data[session_hash]["axis_names"],
            residual_axis=axisMap[which_axis]
        )
        user_data[session_hash]["axis"][axisMap[which_axis]] = residual
        user_data[session_hash]["axis_names"][axisMap[which_axis]] = axis_name
    
    user_data[session_hash]["coords"] = get_concat_embeddings(user_data[session_hash]["examples"]) @ user_data[session_hash]["axis"].T
    user_data[session_hash]["coords"][:, 1] = 5 * (1.0 - user_data[session_hash]["coords"][:, 1])
    
    return update_user_fig(session_hash)

with gr.Blocks(css="#step_size_circular {background-color: #666666} #step_size_circular textarea {background-color: #666666}") as demo:
    gr.Markdown("## Stable Diffusion Embeddings Demo")
    
    session_hash_state = gr.State("")
    
    with gr.TabItem("Embeddings"):
        with gr.Row():
            output = gr.HTML(
                value="Loading...",
                elem_id="embedding-html"
            )
        
        with gr.Row():
            word2add_rem = gr.Textbox(lines=1, label="Add/Remove word")
            word2change = gr.Textbox(lines=1, label="Change image for word")
            clear_words_button = gr.Button(value="Clear words")
        
        with gr.Accordion("Custom Semantic Dimensions", open=False):
            with gr.Row():
                axis_name_1 = gr.Textbox(label="Axis name", value="gender")
                which_axis_1 = gr.Dropdown(
                    choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                    value=whichAxisMap["which_axis_1"],
                    label="Axis direction",
                )
                from_words_1 = gr.Textbox(
                    lines=1,
                    label="Positive",
                    value="prince husband father son uncle",
                )
                to_words_1 = gr.Textbox(
                    lines=1,
                    label="Negative",
                    value="princess wife mother daughter aunt",
                )
                submit_1 = gr.Button("Submit")
        
        def load_user_html(request: gr.Request):
            flask_url, session_hash = init_user_session(request)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            return html_content, session_hash
        
        demo.load(load_user_html, None, [output, session_hash_state])
        
        @word2add_rem.submit(inputs=[word2add_rem, session_hash_state], outputs=[output, word2add_rem])
        def add_rem_word_handler(words, session_hash):
            flask_url = add_rem_word_user(words, session_hash)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            return html_content, ""
        
        @word2change.submit(inputs=[word2change, session_hash_state], outputs=[output, word2change])
        def change_word_handler(word, session_hash):
            flask_url = change_word_user(word, session_hash)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            return html_content, ""
        
        @clear_words_button.click(
            fn=lambda session_hash: (
                f"""<iframe id="html-frame" src="{clear_words_user(session_hash)}" style="width:100%; height:700px;"></iframe>"""
            ),
            inputs=[session_hash_state],
            outputs=[output]
        )
        
        @submit_1.click(
            inputs=[axis_name_1, which_axis_1, from_words_1, to_words_1, session_hash_state],
            outputs=[output],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            whichAxisMap["which_axis_1"] = which_axis
            
            flask_url = set_axis_user(axis_name, which_axis, from_words, to_words, session_hash)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            return html_content

if os.path.exists("user_sessions"):
    for session_dir in os.listdir("user_sessions"):
        try:
            shutil.rmtree(os.path.join("user_sessions", session_dir))
        except:
            pass

if __name__ == "__main__":
    try:
        os.makedirs("outputs", exist_ok=True)
        demo.queue().launch(share=True)
    except KeyboardInterrupt:
        print("Server closed")