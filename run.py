import os
import base64
import gradio as gr
import json
import random
import time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from base import *
from params import *
from embeddings import *
from clip_config import *
from threading import Thread
import plotly.express as px
import numpy as np
from flask import Flask, send_file, request as flask_request
from session import session_manager  

flask_app = Flask(__name__)

session_manager.start_cleanup_thread()

default_examples = examples.copy()
default_images = images.copy()
default_coords = coords.copy()
user_data = {}

def get_safe_filename(word):
    """Convert a word to a safe filename"""
    return "".join([c if c.isalnum() else "_" for c in word])

def get_user_dir(session_hash):
    """Get the main directory for a specific user's session"""
    if not session_hash:
        return None
    user_dir = session_manager.get_session_path(session_hash)
    print(f"User directory path: {user_dir.absolute()}")
    return user_dir

def get_user_examples_dir(session_hash):
    """Get the examples directory for a specific user's session"""
    if not session_hash:
        return None
    examples_dir = session_manager.get_file_path(session_hash, "examples")
    examples_dir.mkdir(exist_ok=True)
    return examples_dir

def get_user_viz_dir(session_hash):
    """Get the visualizations directory for a specific user's session"""
    if not session_hash:
        return None
    viz_dir = session_manager.get_file_path(session_hash, "visualizations")
    viz_dir.mkdir(exist_ok=True)
    return viz_dir

@flask_app.route('/plot/<session_hash>')
def serve_user_plot(session_hash):
    user_dir = get_user_dir(session_hash)
    if not user_dir:
        return "Invalid session", 404
    
    html_path = user_dir / "embedding_plot.html"
    abs_html_path = html_path.absolute()
    str_html_path = str(abs_html_path)
    
    print(f"Trying to serve file at: {str_html_path}")
    print(f"File exists: {html_path.exists()}")
    print(f"Absolute path: {abs_html_path}")
    
    if html_path.exists():
        try:
            return send_file(str_html_path, mimetype='text/html')
        except Exception as e:
            print(f"Error serving file: {e}")
            return f"Error serving file: {e}", 500
    else:
        return f"Plot not found at {str_html_path}", 404

@flask_app.route('/examples/<session_hash>/<image_name>')
def serve_user_example_image(session_hash, image_name):
    examples_dir = get_user_examples_dir(session_hash)
    if not examples_dir:
        return "Invalid session", 404
    
    # Use absolute path
    image_path = examples_dir / image_name
    abs_image_path = image_path.absolute()
    str_image_path = str(abs_image_path)
    
    print(f"Trying to serve image at: {str_image_path}")
    print(f"Image exists: {image_path.exists()}")
    
    if image_path.exists():
        try:
            return send_file(str_image_path, mimetype='image/jpeg')
        except Exception as e:
            print(f"Error serving user example image: {e}")
            return f"Error serving image: {e}", 500
    else:
        return f"Image not found at {str_image_path}", 404

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
    """Generate the HTML file for a user's session"""
    if not session_hash:
        return None
    
    user_dir = get_user_dir(session_hash)
    if not user_dir:
        return None
    
    html_path = user_dir / "embedding_plot.html"
    abs_html_path = html_path.absolute()
    str_html_path = str(abs_html_path)
    
    user_fig = user_data[session_hash]["fig"]
    
    user_fig.write_html(
        str_html_path,
        full_html=True,
        include_plotlyjs='cdn',
        config={'responsive': True}
    )
    
    print(f"Generated HTML at: {str_html_path}")
    print(f"File exists after generation: {html_path.exists()}")
    print(f"Absolute path: {abs_html_path}")
    
    try:
        os.chmod(str_html_path, 0o644)
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
        
        examples_dir = get_user_examples_dir(session_hash)
        
        if examples_dir:
            for example in user_data[session_hash]["examples"]:
                image_path = os.path.join(examples_dir, f"{get_safe_filename(example)}.jpg")
                if not os.path.exists(image_path):
                    try:
                        image = pipe(
                            prompt=example,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                        ).images[0]
                        image.save(image_path, format="JPEG")
                    except Exception as e:
                        print(f"Error generating initial image for '{example}': {e}")
    
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
    
    examples_dir = get_user_examples_dir(session_hash)
    safe_filename = get_safe_filename(new_example)
    image_path = examples_dir / f"{safe_filename}.jpg"
    image.save(str(image_path), format="JPEG")
    
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
    
    examples_dir = get_user_examples_dir(session_hash)
    safe_filename = get_safe_filename(word_to_remove)
    image_path = examples_dir / f"{safe_filename}.jpg"
    if image_path.exists():
        try:
            image_path.unlink()  
        except Exception as e:
            print(f"Warning: Could not remove image file: {e}")
    
    viz_dir = get_user_viz_dir(session_hash)
    viz_path = viz_dir / f"{safe_filename}_emb.png"
    if viz_path.exists():
        try:
            viz_path.unlink()  
        except Exception as e:
            print(f"Warning: Could not remove visualization file: {e}")
    
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

def generate_word_embedding_visualization(word, session_hash):
    """Generate and save the word embedding visualization"""
    if not session_hash or not word:
        return None, None, "Invalid session or word"
    
    try:
        if session_hash not in user_data:
            return None, None, f"Invalid session"
            
        if word not in user_data[session_hash]["examples"]:
            return None, None, f"Error: '{word}' not in examples"
        
        examples_dir = get_user_examples_dir(session_hash)
        viz_dir = get_user_viz_dir(session_hash)
        
        if not examples_dir or not viz_dir:
            return None, None, "Error: Could not create directories"
        
        str_viz_dir = str(viz_dir)
        emb_viz_b64 = generate_word_emb_vis(word, save_to_file=True, viz_dir=str_viz_dir)
        
        emb_viz_bytes = base64.b64decode(emb_viz_b64.split(',')[1])
        emb_viz = Image.open(BytesIO(emb_viz_bytes))
        
        image_path = examples_dir / f"{get_safe_filename(word)}.jpg"
        
        if image_path.exists():
            generated_img = Image.open(str(image_path))
        else:
            image = pipe(
                prompt=word,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            image.save(str(image_path), format="JPEG")
            generated_img = image
        
        return emb_viz, generated_img, f"Visualization for '{word}'"
    except Exception as e:
        print(f"Error generating visualization for '{word}': {e}")
        return None, None, f"Error: {str(e)}"

def load_user_gallery(session_hash):
    """Load the gallery of example images for this user"""
    if not session_hash:
        return []
    
    if session_hash not in user_data:
        return []
    
    examples_dir = get_user_examples_dir(session_hash)
    if not examples_dir:
        return []
    
    example_images = []
    
    for example in user_data[session_hash]["examples"]:
        image_path = examples_dir / f"{get_safe_filename(example)}.jpg"
        
        if not image_path.exists():
            try:
                image = pipe(
                    prompt=example,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
                
                image.save(str(image_path), format="JPEG")
                example_images.append((image, example))
            except Exception as e:
                print(f"Error generating image for '{example}': {e}")
                continue
        else:
            try:
                image = Image.open(str(image_path))
                example_images.append((image, example))
            except Exception as e:
                print(f"Error loading image for '{example}': {e}")
                continue
    
    return example_images

with gr.Blocks(css="#step_size_circular {background-color: #666666} #step_size_circular textarea {background-color: #666666}") as demo:
    gr.Markdown("## Stable Diffusion Embeddings Demo")
    
    session_hash_state = gr.State("")
    
    with gr.Tabs():
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
                    
            with gr.Row():
                with gr.Column(scale=1):
                    word_input = gr.Textbox(
                        label="Visualize embedding for word",
                        lines=1
                    )
                
                with gr.Column(scale=1):
                    embedding_visualization = gr.Image(
                        type="pil",
                        interactive=False,
                        height="6vw"
                    )
            
            with gr.Row():
                gallery = gr.Gallery(
                    label="Images of words",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    height="auto"
                )
    
    def load_user_html(request: gr.Request):
        flask_url, session_hash = init_user_session(request)
        html_content = f"""
        <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
        """
        return html_content, session_hash
    
    demo.load(load_user_html, None, [output, session_hash_state])
    
    @word2add_rem.submit(inputs=[word2add_rem, session_hash_state], outputs=[output, word2add_rem, gallery])
    def add_rem_word_handler(words, session_hash):
        flask_url = add_rem_word_user(words, session_hash)
        html_content = f"""
        <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
        """
        gallery_images = load_user_gallery(session_hash)
        return html_content, "", gallery_images
    
    @word2change.submit(inputs=[word2change, session_hash_state], outputs=[output, word2change, gallery])
    def change_word_handler(word, session_hash):
        flask_url = change_word_user(word, session_hash)
        html_content = f"""
        <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
        """
        gallery_images = load_user_gallery(session_hash)
        return html_content, "", gallery_images
    
    @clear_words_button.click(inputs=[session_hash_state], outputs=[output, gallery])
    def clear_words_handler(session_hash):
        clear_url = clear_words_user(session_hash)
        html_content = f"""<iframe id="html-frame" src="{clear_url}" style="width:100%; height:700px;"></iframe>"""
        gallery_images = load_user_gallery(session_hash)
        return html_content, gallery_images
    
    @submit_1.click(inputs=[axis_name_1, which_axis_1, from_words_1, to_words_1, session_hash_state], outputs=[output, gallery])
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
        whichAxisMap["which_axis_1"] = which_axis
        
        flask_url = set_axis_user(axis_name, which_axis, from_words, to_words, session_hash)
        html_content = f"""
        <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
        """
        gallery_images = load_user_gallery(session_hash)
        return html_content, gallery_images

    @word_input.submit(inputs=[word_input, session_hash_state], outputs=[embedding_visualization, word_input, gallery])
    def handle_word_visualization(word, session_hash):
        if not word.strip():
            return None, None, "Please enter a word", "", load_user_gallery(session_hash)
            
        emb_viz, generated_img, label = generate_word_embedding_visualization(word, session_hash)
        
        if "not in examples" in label:
            gr.Warning(f"'{word}' not in examples. Please add it first using the Add/Remove word field.")
            return None, None, "", load_user_gallery(session_hash)
            
        return emb_viz, "", load_user_gallery(session_hash)
        
    @demo.load(inputs=[session_hash_state], outputs=[gallery])
    def load_gallery_on_start(session_hash):
        if not session_hash:
            return []
        return load_user_gallery(session_hash)


if __name__ == "__main__":
    try:
        os.makedirs("outputs", exist_ok=True)
        demo.queue().launch(share=True)
    except KeyboardInterrupt:
        print("Server closed")
        session_manager.stop_cleanup_thread()