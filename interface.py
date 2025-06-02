#!/usr/bin/env python3
"""
IQC Assistant Interface
"""

import os
import gradio as gr

from agent import ask_agent_sync
from config import LOGO_PATH, INTERFACE_SERVER_URL

logo_path = LOGO_PATH
if not os.path.isfile(logo_path):
    print("⚠️  Warning: logo.png not found in the same folder as this script.")


with gr.Blocks(title="IQC Assistant") as iface:
    if os.path.isfile(logo_path):
        gr.Image(
            value=logo_path,
            interactive=False,
            elem_id="iqc-logo",
            width=150
        )

    gr.Markdown(
        """
        # IQC Assistant  
        An intelligent agent designed for interactive quality control and optimization of seismic data.
        """
    )

    user_input = gr.Textbox(
        label="Your Question",
        placeholder=(
            "Examples:\n"
            "• load file mydata.sgy\n"
            "• plot QC summary\n"
            "• show inline section 150\n"
            "• detect bad traces\n"
            "• adjust thresholds to 0.2 and 2.5"
        ),
        lines=2
    )

    agent_output = gr.Textbox(
        label="Assistant Reply",
        lines=12,
        interactive=False
    )

    submit_btn = gr.Button("Send")
    submit_btn.click(fn=ask_agent_sync, inputs=user_input, outputs=agent_output)

    gr.Markdown(
        """
        Made with ❤️ for the [EAGE Annual 2025 hackathon](https://github.com/EAGE-Annual-Hackathon/EAGE-Hackathon-2025-Instructions/) by Xiaoxuan Zhu and Valentin Cassayre
        """
    )

server_port = int(INTERFACE_SERVER_URL.split(":")[-1])
if __name__ == "__main__":
    iface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=server_port
    )
