import spacy
import benepar
from nltk import Tree
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Add benepar parser
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def save_trees_to_html(doc, filename="trees.html"):

    html_parts = []

    html_parts.append("""
    <html>
    <head>
    <style>
    body {font-family: Arial; margin:40px;}
    .container {display:flex; gap:30px; margin-bottom:50px;}
    .box {width:50%; border:1px solid #ccc; padding:15px;}
    h2 {text-align:center;}
    </style>
    </head>
    <body>
    """)

    for sent in doc.sents:

        html_parts.append(f"<h2>Sentence: {sent.text}</h2>")

        try:
            parse_string = sent._.parse_string
            tree = Tree.fromstring(parse_string)
            constituency_html = "<pre>" + tree.pformat() + "</pre>"
        except:
            constituency_html = "<p>Error generating constituency tree</p>"

        dependency_html = displacy.render(sent, style="dep", page=False)

        html_parts.append(f"""
        <div class="container">
            <div class="box">
                <h3>Constituency Tree</h3>
                {constituency_html}
            </div>

            <div class="box">
                <h3>Dependency Tree</h3>
                {dependency_html}
            </div>
        </div>
        """)

    html_parts.append("</body></html>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print("HTML file created:", filename)


if __name__ == "__main__":

    text = """
    The boy is reading a book.
    What are you doing?
    Close the door.
    How beautiful the sunset is!
    """

    doc = nlp(text)

    save_trees_to_html(doc)