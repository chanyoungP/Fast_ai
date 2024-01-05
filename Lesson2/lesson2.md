# Lesson 2: Deployment 
```
We’ll be using a particular deployment target called Hugging Face Space with Gradio, and will also see how to use JavaScript to implement an interface in the browser
```

### Learning Concept of Hugging Face Spaces and Gradio 

### What is Hugging Face Spaces? 
- It resembles GitHub but offers a convenient space to easily create ML demo apps.
  
### What is Gradio? 
- It allows us to create a web app for deep learning models, similar to Flask.
- However, it's uncertain if it can handle very heavy models or apps.

### Learn How to Make a Web App Step by Step 

#### Step 1: Data Preprocessing 
- Data preprocessing involves image processing techniques such as squishing (making images thinner), padding (zero, etc.), and random resize cropping (data augmentation).
- By applying data augmentation, we can generate different data using one image.

#### Step 2: Training Model 
- Check training results, including accuracy, loss, and confusion matrix (real vs. predicted).
- Perform Hyperparameter Optimization.
- Loss might be high even if the model predicts correctly when it predicts with low confidence.
- **You can have a bad loss either by being wrong and confident or being right and unconfident. Why?**
- Before you start data cleaning, always build a model to find out what things are difficult to recognize in your data and identify issues that the model can help you discover. This will help you address data problems, such as gathering more data or deleting irrelevant data.


- **Now that we've cleaned our data, how are we going to put it into production?**

#### Step 3: Gradio + [Hugging Face Spaces](https://huggingface.co/spaces): A Tutorial (참고)
- Hugging Face Spaces can take the model we trained and copy it to the Hugging Face Spaces server and write a user interface for it.
- **How to use Spaces:**
  1. Create a credit account.
  2. Create Spaces.
  3. Clone the Hugging Face Spaces repository.
  4. Create a Gradio `app.py` file.
  
```python
import gradio as gr

def greet(name):
    return f'Hello {name}!'

gr.Interface(fn=greet, inputs='text', outputs='text').launch()




Make a deep learning model a web app using Hugging  Face Spaces and Gradio:
1 Build a deep learning model.
2 Train the model.
3 Export the model as model.pkl or model.(??).
...

- ??functionname: You can see the source code.
doc(functionname): You can see the documentation of the function.


- %time code: You can check the execution time.


