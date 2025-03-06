import os
import gradio as gr
from clip_config import whichAxisMap
from threading import Thread
from session import session_manager
from serve import run_flask_server
from embeddings import (
    init_user_session,
    load_user_gallery,
    add_rem_word_user,
    change_word_user,
    clear_words_user,
    set_axis_user,
    generate_word_embedding_visualization,
)

with gr.Blocks(
    css="#step_size_circular {background-color: #666666} #step_size_circular textarea {background-color: #666666}"
) as demo:
    gr.Markdown("## Stable Diffusion Embeddings Demo")

    session_hash_state = gr.State("")

    with gr.Tabs():
        with gr.TabItem("Embeddings"):
            with gr.Row():
                output = gr.HTML(value="Loading...", elem_id="embedding-html")

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
                        label="Visualize embedding for word", lines=1
                    )

                with gr.Column(scale=1):
                    embedding_visualization = gr.Image(
                        type="pil", interactive=False, height="6vw"
                    )

            with gr.Row():
                gallery = gr.Gallery(
                    label="Images of words",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )

    def load_user_html(request: gr.Request):
        flask_url, session_hash, is_new = init_user_session(request)
        html_content = f"""
        <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
        """
        if is_new:
            gr.Info("New session initialized.")

        gallery_images = load_user_gallery(session_hash)

        return html_content, session_hash, gallery_images

    demo.load(load_user_html, None, [output, session_hash_state, gallery])

    @word2add_rem.submit(
        inputs=[word2add_rem, session_hash_state],
        outputs=[output, word2add_rem, gallery],
    )
    def add_rem_word_handler(words, session_hash):
        flask_url = add_rem_word_user(words, session_hash)
        html_content = f"""
        <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
        """
        gallery_images = load_user_gallery(session_hash)
        return html_content, "", gallery_images

    @word2change.submit(
        inputs=[word2change, session_hash_state], outputs=[output, word2change, gallery]
    )
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

    @submit_1.click(
        inputs=[
            axis_name_1,
            which_axis_1,
            from_words_1,
            to_words_1,
            session_hash_state,
        ],
        outputs=[output, gallery],
    )
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
        whichAxisMap["which_axis_1"] = which_axis

        flask_url = set_axis_user(
            axis_name, which_axis, from_words, to_words, session_hash
        )
        html_content = f"""
        <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
        """
        gallery_images = load_user_gallery(session_hash)
        return html_content, gallery_images

    @word_input.submit(
        inputs=[word_input, session_hash_state],
        outputs=[embedding_visualization, word_input, gallery],
    )
    def handle_word_visualization(word, session_hash):
        if not word.strip():
            return None, "", load_user_gallery(session_hash)

        emb_viz, generated_img, label = generate_word_embedding_visualization(
            word, session_hash
        )

        if "not in examples" in label:
            gr.Warning(
                f"'{word}' not in examples. Please add it first using the Add/Remove word field."
            )
            return None, "", load_user_gallery(session_hash)

        return emb_viz, "", load_user_gallery(session_hash)


if __name__ == "__main__":
    try:
        flask_thread = Thread(target=run_flask_server)
        flask_thread.daemon = True
        flask_thread.start()

        session_manager.start_cleanup_thread()

        os.makedirs("outputs", exist_ok=True)
        demo.queue().launch(share=True)
    except KeyboardInterrupt:
        print("Server closed")
        session_manager.stop_cleanup_thread()
