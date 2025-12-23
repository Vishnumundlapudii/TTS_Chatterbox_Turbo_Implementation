# Chatterbox Turbo TTS Demo Setup Guide

## 🎯 Overview
This guide helps you set up and run the Chatterbox Turbo TTS demo on a GPU-enabled server. The demo provides a web interface for text-to-speech generation with voice cloning capabilities.

## 📋 Requirements

### Hardware
- **GPU**: Minimum 16GB VRAM (Recommended: 24GB+ for optimal performance)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space

### Software
- **OS**: Ubuntu 20.04+ or 22.04 LTS (recommended)
- **CUDA**: Version 11.8 or 12.1
- **Python**: 3.9-3.11 (3.10 recommended)
  
### Deployment Environment
  This implementation was successfully tested and deployed on [Jarvislabs.ai](https://jarvislabs.ai/) with:
  - **GPU:** A5000 24GB VRAM
  - **CUDA:** Support enabled
  - **OS:** Ubuntu environment
  - **Python:** 3.10
  
## 🚀 Step-by-Step Setup

### 1. Initial System Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git wget curl htop nvtop build-essential

# Verify GPU and CUDA
nvidia-smi
```

### 2. Install Miniconda
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

# Add to PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Initialize conda
conda init bash
source ~/.bashrc
```

### 3. Create Python Environment
```bash
# Create new environment with Python 3.10
conda create -n chatterbox python=3.10 -y
conda activate chatterbox

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. Install Chatterbox TTS
```bash
# Install the main package
pip install chatterbox-tts

# Install additional dependencies
pip install streamlit soundfile librosa numpy scipy
```

### 5. Create Demo Directory and Files
```bash
# Create demo directory
mkdir ~/chatterbox_demo
cd ~/chatterbox_demo
```

Create the demo script file:
```bash
import streamlit as st
import torchaudio as ta
import torch
import numpy as np
import soundfile as sf
import tempfile
import os
from chatterbox.tts_turbo import ChatterboxTurboTTS

@st.cache_resource
def load_model():
    """Load the Chatterbox TTS model (cached to avoid reloading)"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def create_silence_reference(duration=10, sample_rate=22050):
    """Create a silent reference audio file"""
    silence = np.zeros(duration * sample_rate, dtype=np.float32)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, silence, sample_rate)
    return temp_file.name

def main():
    st.set_page_config(
        page_title="Chatterbox Turbo TTS Demo",
        page_icon="🎤",
        layout="wide"
    )

    st.title("🎤 Chatterbox Turbo TTS Demo")
    st.markdown("### Generate realistic speech with paralinguistic tags")

    # Load model
    model, device = load_model()
    if model is None:
        st.error("Failed to load model. Please check Hugging Face authentication.")
        st.info("Run: `huggingface-cli login` in terminal to authenticate.")
        return

    st.success(f"✅ Model loaded successfully on {device}")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Voice cloning option
    use_voice_cloning = st.sidebar.checkbox("Use Voice Cloning", value=False)

    uploaded_file = None
    if use_voice_cloning:
        uploaded_file = st.sidebar.file_uploader(
            "Upload reference audio (WAV, 5+ seconds)",
            type=['wav'],
            help="Upload an audio file to clone the voice from"
        )

    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1)
    top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9, 0.05)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Text Input")

        # Example texts
        example_texts = {
            "Customer Service": "Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat about the billing issue?",
            "Excited Announcement": "Hey everyone! [excited] I'm absolutely thrilled to announce our new product launch [pause] this is going to be amazing!",
            "Thoughtful Explanation": "So [pause] the way this works is quite interesting [thoughtful]. Let me break it down for you step by step.",
            "Casual Conversation": "Oh hey! [laugh] I totally forgot about that meeting. Thanks for reminding me [relieved]."
        }

        selected_example = st.selectbox("Choose an example:", ["Custom"] + list(example_texts.keys()))

        if selected_example != "Custom":
            text_input = st.text_area(
                "Text to generate:",
                value=example_texts[selected_example],
                height=100
            )
        else:
            text_input = st.text_area(
                "Text to generate:",
                placeholder="Enter your text here. Use tags like [chuckle], [pause], [excited], [laugh], etc.",
                height=100
            )

        # Paralinguistic tags help
        with st.expander("ℹ️ Available Paralinguistic Tags"):
            st.markdown("""
            **Emotional Tags:**
            - `[excited]`, `[happy]`, `[sad]`, `[angry]`, `[surprised]`
            - `[thoughtful]`, `[relieved]`, `[confused]`, `[worried]`

            **Action Tags:**
            - `[laugh]`, `[chuckle]`, `[giggle]`, `[sigh]`, `[gasp]`
            - `[pause]`, `[breath]`, `[whisper]`, `[shout]`

            **Example:** "Hello there [pause] how are you doing today [chuckle]?"
            """)

    with col2:
        st.subheader("🎵 Audio Reference")

        if use_voice_cloning and uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            st.success("✅ Voice reference uploaded")
        elif use_voice_cloning:
            st.info("📤 Please upload a voice reference file")
        else:
            st.info("🔊 Using default voice (no cloning)")

    # Generation section
    st.markdown("---")

    if st.button("🎤 Generate Speech", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("Please enter some text to generate!")
            return

        try:
            with st.spinner("Generating speech... This may take a moment."):
                # Handle audio reference
                audio_prompt_path = None

                if use_voice_cloning and uploaded_file:
                    # Save uploaded file temporarily
                    temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_ref.write(uploaded_file.read())
                    temp_ref.close()
                    audio_prompt_path = temp_ref.name
                elif use_voice_cloning:
                    st.warning("Voice cloning enabled but no file uploaded. Using silence reference.")
                    audio_prompt_path = create_silence_reference()

                # Generate audio
                if audio_prompt_path:
                    wav = model.generate(
                        text_input,
                        audio_prompt_path=audio_prompt_path,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p
                    )
                else:
                    wav = model.generate(
                        text_input,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p
                    )

                # Save generated audio
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                ta.save(output_path.name, wav, model.sr)

                # Display results
                st.success("🎉 Speech generated successfully!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔊 Generated Audio")
                    st.audio(output_path.name, format="audio/wav")

                with col2:
                    st.subheader("📊 Audio Info")
                    duration = len(wav[0]) / model.sr
                    st.metric("Duration", f"{duration:.2f}s")
                    st.metric("Sample Rate", f"{model.sr} Hz")
                    st.metric("Channels", wav.shape[0])

                # Download button
                with open(output_path.name, "rb") as f:
                    st.download_button(
                        label="📥 Download Audio",
                        data=f.read(),
                        file_name="generated_speech.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

                # Cleanup temporary files
                if audio_prompt_path:
                    try:
                        os.unlink(audio_prompt_path)
                    except:
                        pass

        except Exception as e:
            st.error(f"Error generating speech: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
```

### 6. Hugging Face Authentication
```bash
# Install Hugging Face CLI
pip install --upgrade huggingface_hub

# Login to Hugging Face (you'll need a token)
huggingface-cli login
```

**To get your Hugging Face token:**
1. Visit: https://huggingface.co/settings/tokens
2. Create a new token with "Read" access
3. Copy the token and paste it when prompted

### 7. Test Installation
```bash
# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Chatterbox import
python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; print('✅ Import successful')"
```

### 8. Run the Demo
```bash
# Start the Streamlit demo
streamlit run demo.py --server.port 8501 --server.address 0.0.0.0
```

## 🌐 Accessing the Demo

### Local Access
- Open browser and go to: `http://localhost:8501`

### Remote Access
- Access via: `http://YOUR_SERVER_IP:8501`
- **Security Note**: Make sure port 8501 is allowed in your firewall

### Secure Access (SSH Tunnel)
```bash
# On your local machine, create SSH tunnel
ssh -L 8501:localhost:8501 username@your-server-ip

# Then access via: http://localhost:8501
```

## 🎮 Using the Demo

### Basic Text-to-Speech
1. Enter text in the text area
2. Click "Generate Speech"
3. Listen to the generated audio
4. Download if needed

### Voice Cloning
1. Enable "Use Voice Cloning" in sidebar
2. Upload a reference audio file (5+ seconds, WAV format)
3. Enter your text
4. Generate speech with cloned voice

### Paralinguistic Tags
Use special tags in your text to add emotions and actions:
- `[chuckle]`, `[laugh]`, `[giggle]` for laughter
- `[pause]`, `[breath]` for timing
- `[excited]`, `[happy]`, `[sad]` for emotions

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size or use smaller text chunks
- Restart the application

**2. Model Loading Fails**
- Check Hugging Face authentication: `huggingface-cli whoami`
- Verify internet connection

**3. Audio Upload Issues**
- Ensure audio file is WAV format
- Check file is at least 5 seconds long

**4. Port Already in Use**
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use different port
streamlit run demo.py --server.port 8502
```

### Performance Tips
- First model load takes ~2-5 minutes (downloading weights)
- Subsequent generations are much faster
- Keep the browser tab active for better performance

## 📞 Support

For technical issues:
1. Check the troubleshooting section above
2. Verify all requirements are met
3. Check server logs for detailed error messages

## 📄 License

This demo uses the Chatterbox TTS model from ResembleAI. Please check their license terms for commercial usage.

---

**Happy Testing! 🎤🚀**
