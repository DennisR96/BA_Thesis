
import gradio as gr


def greet(name):
    return "Hello " + name + "!"


full = gr.Interface(
  fn=greet, 
  inputs="text", 
  outputs="text"
  )

inpaint = gr.Interface(
  fn=greet, 
  inputs="text", 
  outputs="text"
  )

resen = gr.Interface(
  fn=greet, 
  inputs="text", 
  outputs="text"
  )

fegen = gr.Interface(
  fn=greet, 
  inputs="text", 
  outputs="text"
  )

demo = gr.TabbedInterface([full, inpaint, resen, fegen], ["1. Full-Model", "2. In-Painting","3. Object Detection","4. Classification"])

demo.launch(share=True)





