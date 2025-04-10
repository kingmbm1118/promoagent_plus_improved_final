import os
import shutil
import streamlit as st
import tempfile
from dotenv import load_dotenv

from promoagent_plus.main import ProMoAgentPlus
from promoagent_plus.utils.constants import InputType, ViewType, AIProviders, AI_MODEL_DEFAULTS, DEFAULT_AI_PROVIDER
import pm4py
from pm4py.util import constants
from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.objects.bpmn.layout import layouter as bpmn_layouter
from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string

# Load environment variables
load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = "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"

# Update default AI provider to OpenAI
DEFAULT_AI_PROVIDER = AIProviders.OPENAI.value

# Update the default model for OpenAI to gpt-4o-mini, and keep only models that support agentic LLM or CrewAI
AI_MODEL_DEFAULTS = {
    AIProviders.GOOGLE.value: "gemini-1.5-pro",
    AIProviders.OPENAI.value: "gpt-4o-mini",
    AIProviders.ANTHROPIC.value: "claude-3-opus-20240229"
}

# Help text constants
MAIN_HELP = "Select the AI provider you'd like to use. Google offers the Gemini models,"\
            " which you can **try for free**,"\
            " while OpenAI provides GPT models. Anthropic provides Claude models."\
            " All selected providers support agentic capabilities required for multi-agent process modeling."

AI_HELP_DEFAULTS = {
    AIProviders.GOOGLE.value: "Enter a Google model name that supports agentic capabilities (e.g., gemini-1.5-pro). You can get a **free Google API key** and check the latest models under: https://ai.google.dev/.",
    AIProviders.OPENAI.value: "Enter an OpenAI model name that supports agentic capabilities and function calling (e.g., gpt-4o-mini, gpt-4o, gpt-4). You can get an OpenAI API key and check the latest models under: https://openai.com/pricing.",
    AIProviders.ANTHROPIC.value: "Enter an Anthropic model name that supports agentic capabilities (e.g., claude-3-opus, claude-3-sonnet). You can get an Anthropic API key and check the latest models under: https://www.anthropic.com/api.",
}

DISCOVERY_HELP = "The event log will be used to generate a process model using the POWL miner."

def run_app():
    st.title('🤖 ProMoAgent+')

    st.subheader(
        "Process Modeling with Multi-Agent Systems"
    )

    temp_dir = "temp"

    if 'provider' not in st.session_state:
        st.session_state['provider'] = DEFAULT_AI_PROVIDER

    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = AI_MODEL_DEFAULTS[st.session_state['provider']]

    def update_model_name():
        st.session_state['model_name'] = AI_MODEL_DEFAULTS[st.session_state['provider']]

    with st.expander("🔧 Configuration", expanded=True):
        provider = st.radio(
            "Choose AI Provider:",
            options=[AIProviders.GOOGLE.value, AIProviders.OPENAI.value, AIProviders.ANTHROPIC.value],
            index=[AIProviders.GOOGLE.value, AIProviders.OPENAI.value, AIProviders.ANTHROPIC.value].index(DEFAULT_AI_PROVIDER),
            horizontal=True,
            help=MAIN_HELP,
            on_change=update_model_name,
            key='provider',
        )

        if 'model_name' not in st.session_state or st.session_state['provider'] != provider:
            st.session_state['model_name'] = AI_MODEL_DEFAULTS[provider]

        col1, col2 = st.columns(2)
        with col1:
            ai_model_name = st.text_input("Enter the AI model name:",
                                          key='model_name',
                                          help=AI_HELP_DEFAULTS[st.session_state['provider']])
        with col2:
            api_key = st.text_input("API key:", type="password", 
                                    value=os.getenv(f"{provider.upper().replace(' ', '_')}_API_KEY", ""))

    if 'selected_mode' not in st.session_state:
        st.session_state['selected_mode'] = "Model Generation"

    input_type = st.radio("Select Input Type:",
                          options=[InputType.TEXT.value, InputType.MODEL.value, InputType.DATA.value], horizontal=True)

    if input_type != st.session_state['selected_mode']:
        st.session_state['selected_mode'] = input_type
        st.session_state['model_gen'] = None
        st.session_state['feedback'] = []
        st.rerun()

    with st.form(key='model_gen_form'):
        if input_type == InputType.TEXT.value:
            description = st.text_area("For **process modeling**, enter the process description:")
            submit_button = st.form_submit_button(label='Run')
            if submit_button:
                with st.spinner("Generating process model..."):
                    try:
                        # Initialize ProMoAgent+
                        promo_agent = ProMoAgentPlus(
                            api_key=api_key,
                            ai_provider=provider,
                            model_name=ai_model_name
                        )
                        
                        # Generate model from text
                        process_model = promo_agent.generate_model_from_text(description)

                        st.session_state['model_gen'] = process_model
                        st.session_state['feedback'] = []
                    except Exception as e:
                        st.error(body=str(e), icon="⚠️")
                        return

        elif input_type == InputType.DATA.value:
            uploaded_log = st.file_uploader("For **process model discovery**, upload an event log:",
                                            type=["xes", "xes.gz"],
                                            help=DISCOVERY_HELP)
            submit_button = st.form_submit_button(label='Run')
            if submit_button:
                if uploaded_log is None:
                    st.error(body="No file is selected!", icon="⚠️")
                    return
                with st.spinner("Discovering process model from event log..."):
                    try:
                        contents = uploaded_log.read()
                        os.makedirs(temp_dir, exist_ok=True)
                        with tempfile.NamedTemporaryFile(mode="wb", delete=False,
                                                        dir=temp_dir, suffix=uploaded_log.name) as temp_file:
                            temp_file.write(contents)
                            log_path = temp_file.name
                        
                        # Initialize ProMoAgent+
                        promo_agent = ProMoAgentPlus(
                            api_key=api_key,
                            ai_provider=provider,
                            model_name=ai_model_name
                        )
                        
                        # Generate model from event log
                        process_model = promo_agent.generate_model_from_event_log(log_path)
                        
                        shutil.rmtree(temp_dir, ignore_errors=True)

                        st.session_state['model_gen'] = process_model
                        st.session_state['feedback'] = []
                    except Exception as e:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        st.error(body=f"Error during discovery: {e}", icon="⚠️")
                        return
                        
        elif input_type == InputType.MODEL.value:
            uploaded_file = st.file_uploader(
                "For **process model improvement**, upload a semi-block-structured BPMN or Petri net:",
                type=["bpmn", "pnml"]
            )
            submit_button = st.form_submit_button(label='Upload')
            if submit_button:
                if uploaded_file is None:
                    st.error(body="No file is selected!", icon="⚠️")
                    return
                with st.spinner("Processing model file..."):
                    try:
                        file_extension = uploaded_file.name.split(".")[-1].lower()

                        # Initialize ProMoAgent+
                        promo_agent = ProMoAgentPlus(
                            api_key=api_key,
                            ai_provider=provider,
                            model_name=ai_model_name
                        )

                        if file_extension == "bpmn":
                            contents = uploaded_file.read()

                            os.makedirs(temp_dir, exist_ok=True)
                            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bpmn",
                                                            dir=temp_dir) as temp_file:
                                temp_file.write(contents)
                                bpmn_path = temp_file.name
                                
                            # Generate model from BPMN
                            process_model = promo_agent.generate_model_from_bpmn(bpmn_path)
                            shutil.rmtree(temp_dir, ignore_errors=True)

                        elif file_extension == "pnml":
                            contents = uploaded_file.read()

                            os.makedirs(temp_dir, exist_ok=True)
                            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pnml",
                                                            dir=temp_dir) as temp_file:
                                temp_file.write(contents)
                                pnml_path = temp_file.name
                                
                            # Generate model from Petri net
                            process_model = promo_agent.generate_model_from_petri_net(pnml_path)
                            shutil.rmtree(temp_dir, ignore_errors=True)

                        else:
                            st.error(body=f"Unsupported file format {file_extension}!", icon="⚠️")
                            return

                        st.session_state['model_gen'] = process_model
                        st.session_state['feedback'] = []
                        
                    except Exception as e:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        st.error(body=f"Error processing model: {str(e)}", icon="⚠️")
                        return

    if 'model_gen' in st.session_state and st.session_state['model_gen']:

        st.success("Model generated successfully!", icon="🎉")

        col1, col2 = st.columns(2)

        try:
            with col1:
                with st.form(key='feedback_form'):
                    feedback = st.text_area("Feedback:", value="")
                    if st.form_submit_button(label='Update Model'):
                        with st.spinner("Updating model based on feedback..."):
                            try:
                                process_model = st.session_state['model_gen']
                                process_model.update(feedback)
                                st.session_state['model_gen'] = process_model
                            except Exception as e:
                                raise Exception("Update failed! " + str(e))
                            st.session_state['feedback'].append(feedback)

                    if len(st.session_state['feedback']) > 0:
                        with st.expander("Feedback History", expanded=True):
                            i = 0
                            for f in st.session_state['feedback']:
                                i = i + 1
                                st.write("[" + str(i) + "] " + f + "\n\n")

            with col2:
                st.write("Export Model")
                process_model_obj = st.session_state['model_gen']
                powl = process_model_obj.get_powl()
                pn, im, fm = pm4py.convert_to_petri_net(powl)
                bpmn = pm4py.convert_to_bpmn(pn, im, fm)
                bpmn = bpmn_layouter.apply(bpmn)

                download_1, download_2 = st.columns(2)
                with download_1:
                    bpmn_data = get_xml_string(bpmn,
                                               parameters={"encoding": constants.DEFAULT_ENCODING})
                    st.download_button(
                        label="Download BPMN",
                        data=bpmn_data,
                        file_name="process_model.bpmn",
                        mime="application/xml"
                    )

                with download_2:
                    pn_data = export_petri_as_string(pn, im, fm)
                    st.download_button(
                        label="Download PNML",
                        data=pn_data,
                        file_name="process_model.pnml",
                        mime="application/xml"
                    )

            view_option = st.selectbox("Select a view:", [ViewType.BPMN.value, ViewType.POWL.value, ViewType.PETRI.value])

            image_format = str("svg").lower()
            if view_option == ViewType.POWL.value:
                from pm4py.visualization.powl import visualizer
                vis_str = visualizer.apply(powl,
                                           parameters={'format': image_format})

            elif view_option == ViewType.PETRI.value:
                visualization = pn_visualizer.apply(pn, im, fm,
                                                    parameters={'format': image_format})
                vis_str = visualization.pipe(format='svg').decode('utf-8')
            else:  # BPMN
                from pm4py.objects.bpmn.layout import layouter
                layouted_bpmn = layouter.apply(bpmn)
                visualization = bpmn_visualizer.apply(layouted_bpmn,
                                                      parameters={'format': image_format})
                vis_str = visualization.pipe(format='svg').decode('utf-8')

            with st.expander("View Model", expanded=True):
                st.image(vis_str)

        except Exception as e:
            st.error(icon='⚠️', body=str(e))


def footer():
    style = """
        <style>
          .footer-container { 
              position: fixed;
              left: 0;
              bottom: 0;
              width: 100%;
              text-align: center;
              padding: 15px 0;
              background-color: white;
              border-top: 2px solid lightgrey;
              z-index: 100;
          }

          .footer-text, .header-text {
              margin: 0;
              padding: 0;
          }
          .footer-links {
              margin: 0;
              padding: 0;
          }
          .footer-links a {
              margin: 0 10px;
              text-decoration: none;
              color: blue;
          }
          .footer-links img {
              vertical-align: middle;
          }
        </style>
        """

    foot = f"""
        <div class='footer-container'>
            <div class='footer-text'>
                ProMoAgent+ - A Multi-Agent Reimplementation of ProMoAI
            </div>
            <div class='footer-links'>
                <a href="https://github.com/yourusername/promoagent-plus" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-gray?logo=github&logoColor=white&labelColor=black" alt="GitHub Repository">
                </a>
            </div>
        </div>
        """

    st.markdown(style, unsafe_allow_html=True)
    st.markdown(foot, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="ProMoAgent+",
        page_icon="🤖"
    )
    footer()
    run_app()