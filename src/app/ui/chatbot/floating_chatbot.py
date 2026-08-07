"""
Floating chatbot component for DataKit.
Fully functional with Streamlit's session state.
"""

import streamlit as st
from pathlib import Path


def render_floating_chatbot():
    """
    Render a floating chatbot bubble that appears on all pages.
    Uses Streamlit's native components for full functionality.
    """
    
    # === INITIALIZATION ===
    _init_chatbot_state()
    
    # === CSS ===
    st.markdown("""
    <style>
    /* Floating container */
    .floating-chatbot {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    
    /* Chat window */
    .chat-window {
        width: 380px;
        max-width: 90vw;
        height: 480px;
        max-height: 70vh;
        background: white;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        display: none;
        flex-direction: column;
        overflow: hidden;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .chat-window.open {
        display: flex;
        animation: slideUp 0.3s ease;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    /* Header */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
    }
    
    .chat-header-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .status-dot.online { background: #4ade80; }
    .status-dot.offline { background: #f87171; }
    
    .chat-close {
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        opacity: 0.8;
    }
    
    .chat-close:hover { opacity: 1; }
    
    /* Messages */
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 12px 16px;
        background: #f8fafc;
        display: flex;
        flex-direction: column;
        gap: 6px;
        min-height: 0;
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 4px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 2px;
    }
    
    /* Input */
    .chat-input-container {
        display: flex;
        padding: 8px 12px;
        background: white;
        border-top: 1px solid #e2e8f0;
        gap: 8px;
        flex-shrink: 0;
    }
    
    .chat-input-container input {
        flex: 1;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 13px;
        outline: none;
    }
    
    .chat-input-container input:focus {
        border-color: #667eea;
    }
    
    .chat-input-container button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        cursor: pointer;
        font-size: 14px;
        flex-shrink: 0;
    }
    
    .chat-input-container button:hover {
        transform: scale(1.05);
    }
    
    .chat-input-container button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
    }
    
    /* Bubble button */
    .chat-bubble {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        font-size: 28px;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chat-bubble:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 30px rgba(102, 126, 234, 0.6);
    }
    
    .chat-bubble.active {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        transform: rotate(90deg);
    }
    
    /* Message styles */
    .message {
        max-width: 85%;
        padding: 8px 14px;
        border-radius: 12px;
        font-size: 13px;
        line-height: 1.5;
        word-wrap: break-word;
        animation: messageIn 0.3s ease;
    }
    
    @keyframes messageIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message.user {
        align-self: flex-end;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message.assistant {
        align-self: flex-start;
        background: white;
        color: #1e293b;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px;
    }
    
    .message .sources {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid #f1f5f9;
    }
    
    /* Quick suggestions */
    .quick-suggestions {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        padding: 6px 14px 10px;
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        flex-shrink: 0;
    }
    
    .quick-suggestions button {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 11px;
        cursor: pointer;
        color: #475569;
        transition: all 0.2s;
    }
    
    .quick-suggestions button:hover {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    @media (max-width: 640px) {
        .chat-window {
            width: 100vw;
            max-width: 100vw;
            height: 70vh;
            bottom: 80px;
            right: 0;
            border-radius: 16px 16px 0 0;
        }
        .chat-bubble {
            width: 56px;
            height: 56px;
            font-size: 24px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # === GET ASSISTANT ===
    assistant = _get_assistant()
    agent = assistant.get("agent") if assistant else None
    llm_available = assistant.get("llm_client", {}).is_available if assistant else False

    # === STATE ===
    is_open = st.session_state.chatbot_open
    messages = st.session_state.chat_history

    # === RENDER ===
    status_dot = "online" if llm_available else "offline"
    status_text = "En ligne" if llm_available else "Hors ligne"

    # Build messages HTML
    messages_html = ""
    for msg in messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"]
        messages_html += f'<div class="message {role_class}">{content}</div>'

    # Suggestions
    suggestions = [
        "📊 Résumé dataset",
        "🔍 Valeurs manquantes",
        "📈 Visualisation",
        "🤖 Modèle suggéré"
    ]
    suggestions_html = "".join([
        f'<button onclick="document.getElementById(\'chatbot-input\').value=\'{s}\'; document.getElementById(\'chatbot-send\').click();">{s}</button>'
        for s in suggestions
    ])

    # Full HTML
    html = f"""
    <div class="floating-chatbot">
        <!-- Chat Window -->
        <div class="chat-window {'open' if is_open else ''}" id="chat-window">
            <div class="chat-header">
                <div class="chat-header-title">
                    <span class="status-dot {status_dot}"></span>
                    🤖 Assistant DataKit
                    <span style="font-size:12px;opacity:0.8;">{status_text}</span>
                </div>
                <button class="chat-close" onclick="toggleChat()">✕</button>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                {messages_html}
            </div>
            
            <div class="quick-suggestions">
                {suggestions_html}
            </div>
            
            <div class="chat-input-container">
                <input 
                    id="chatbot-input" 
                    type="text" 
                    placeholder="Posez votre question..." 
                    onkeydown="if(event.key==='Enter'){{document.getElementById('chatbot-send').click()}}"
                />
                <button id="chatbot-send" onclick="sendMessage()">➤</button>
            </div>
        </div>
        
        <!-- Bubble -->
        <button class="chat-bubble {'active' if is_open else ''}" onclick="toggleChat()">
            {'✕' if is_open else '💬'}
        </button>
    </div>
    
    <script>
    function toggleChat() {{
        var window = document.getElementById('chat-window');
        var bubble = document.querySelector('.chat-bubble');
        window.classList.toggle('open');
        bubble.classList.toggle('active');
        if (window.classList.contains('open')) {{
            document.getElementById('chatbot-input').focus();
        }}
    }}
    
    function sendMessage() {{
        var input = document.getElementById('chatbot-input');
        var message = input.value.trim();
        if (!message) return;
        input.value = '';
        
        // Send to Streamlit via component
        var event = new CustomEvent('chatbot_message', {{ detail: message }});
        window.dispatchEvent(event);
    }}
    </script>
    """

    st.markdown(html, unsafe_allow_html=True)

    # === HANDLE MESSAGES ===
    _handle_chatbot_input(agent)


def _init_chatbot_state():
    """Initialize chatbot session state."""
    if "chatbot_open" not in st.session_state:
        st.session_state.chatbot_open = False
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "👋 Bonjour ! Je suis l'assistant IA de DataKit. Comment puis-je vous aider ?"}
        ]


def _get_assistant():
    """Get or create the shared assistant."""
    if "assistant" not in st.session_state:
        try:
            from datakit.ai_assistant import create_assistant
            
            knowledge_base_path = None
            default_path = Path("knowledge_base")
            if default_path.exists():
                knowledge_base_path = str(default_path)
            
            st.session_state.assistant = create_assistant(
                model_name="mistral",
                knowledge_base_path=knowledge_base_path
            )
        except Exception as e:
            st.session_state.assistant = None
            st.error(f"❌ Erreur: {e}")
    
    return st.session_state.assistant


def _handle_chatbot_input(agent):
    """
    Handle chatbot input using Streamlit's form.
    Uses a hidden form to capture messages.
    """
    
    if agent is None:
        return
    
    # Hidden form for message capture
    with st.form(key="chatbot_form", clear_on_submit=True):
        prompt = st.text_input(
            "Message",
            key="chatbot_prompt",
            label_visibility="collapsed",
            placeholder="Posez votre question..."
        )
        submitted = st.form_submit_button("Envoyer", type="primary")
    
    if submitted and prompt:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🤔"):
            try:
                response = agent.ask(prompt)
                answer = response.get("answer", "Erreur")
                
                # Add sources if available
                sources = response.get("documents", [])
                if sources:
                    answer += "\n\n📚 **Sources:**"
                    for doc in sources[:3]:
                        source = doc.get("source", "unknown")
                        answer += f"\n- {source}"
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ Erreur: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        st.rerun()