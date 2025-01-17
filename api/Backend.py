from flask import Flask, request, jsonify
import PromptMaker
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    req_data = request.get_json(),
    question = req_data.get("question", "What is Task Decomposition?")
    response = PromptMaker.graphInstance.graph.invoke({"question": question})
    return jsonify({"answer": response["answer"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

@app.route("/")