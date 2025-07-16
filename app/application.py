from app.components.retriever import create_qa_chain

from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for
import os
from app.common.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)


from markupsafe import Markup
def nl2br(value):
    """Convert newlines to <br> tags."""
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/", methods=['GET', 'POST'])
def index():
    if "messages" not in session:
        session["messages"] = []
    if request.method == 'POST':
        user_input = request.form.get("prompt")
        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages
            try:
                qa_chain = create_qa_chain()
                response = qa_chain.invoke({"query": user_input})
                response = response.get("result", "No answer found.")
                messages.append({"role": "assistant", "content": response})
                session["messages"] = messages

            except Exception as e:
                error_message = f"Error : {str(e)}"
                return render_template('index.html', messages=session["messages"], error=error_message)
        return redirect(url_for('index'))
    return render_template('index.html', messages=session.get("messages", []))

@app.route('/clear')
def clear():
    session.pop("messages", None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5000, debug=False, use_reloader=False)
