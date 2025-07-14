### 1. **What is Jupyter?**

Jupyter is an open-source interactive computing environment. Its most well-known tool is the **Jupyter Notebook**, a web-based tool that makes it easy to write code, run code, take notes, visualize data, write math formulas, and more.

In simple terms, a Jupyter Notebook is like an “interactive document.” You can write Python (or even other languages) code directly in your browser, run it instantly to see the results, add explanatory text, insert images, and so on. It’s very well-suited for data science, machine learning, research, and teaching.

---

### 2. How to Use Jupyter?

* **Installation**: The easiest way is to install Anaconda (which comes with Jupyter), or install it directly via pip:

  ```bash
  pip install notebook
  ```

  or:

  ```bash
  pip install jupyter
  ```

* **Launch**: Run the following in your terminal (command line):

  ```
  jupyter notebook
  ```

  This will:

  * Automatically open your browser
  * Show a file browser-like interface

  If your browser doesn’t open automatically, it usually displays a URL such as:

  ```
  http://localhost:8888/tree
  ```

  You can copy this link into your browser manually.

## Creating a New Notebook

In the Jupyter web page:

* Click “New” at the top right
* Choose **Python 3** (or any other kernel you’ve installed)

This creates a new blank Notebook.

---

## Writing Code in a Notebook

Each gray block is a **cell**. You can:

* Write Python code in a cell
* Press Shift + Enter to run it
* The output appears directly below the cell

Example:

```python
print("Hello, Jupyter!")
```

After running it, you’ll see:

```
Hello, Jupyter!
```

---

## Writing Text Notes

Jupyter isn’t just for writing code—you can also write text, equations, and charts.

In a cell:

* Click the dropdown that says “Code” → change it to “Markdown”
* Now you can write text, headings, lists, equations, etc.

Example:

```
# My Notes

This is a Jupyter Notebook.

- Supports lists
- Supports equations: $E=mc^2$
```

When you run it, it turns into nicely formatted text.

---

## Saving Your Work

Jupyter Notebooks are saved as `.ipynb` files.

* You can go to File → Save
* Or just press Ctrl+S
* Next time, open it again with `jupyter notebook` and continue working

---

## Closing Jupyter

* Simply close the browser tab
* Go back to the terminal and press `Ctrl+C` to stop the Jupyter server

---

### Common Keyboard Shortcuts

* **Shift + Enter** → Run current cell and move to the next
* **Ctrl + Enter** → Run current cell but stay in place
* **Esc** → Switch to command mode
* **A** → Insert a cell above
* **B** → Insert a cell below
* **D + D** (press D twice) → Delete cell

---

# Exporting to a `.py` Script Using nbconvert

The most universal way is via the **nbconvert** command line tool.

Suppose you have:

```
Test.ipynb
```

In the terminal, run:

```bash
jupyter nbconvert --to script Test.ipynb
```

After running this, you’ll see:

```
Test.py
```

in the same directory.

Example:

```bash
jupyter nbconvert --to script Test.ipynb
```

## Running the Python File Directly in the Terminal

In your terminal, type:

```bash
python Test.py
```

This command will:

* Run your entire script from top to bottom
* It’s the same as clicking “Run All” in Jupyter Notebook

You should see something like:

```
Epoch 1 - Loss: 0.6xxxx
Epoch 2 - Loss: 0.3xxxx
...
```

And eventually, a loss curve graph will pop up.

---

```bash
python generate_data.py
python train_real_data.py
```

## Installing Extensions

If you’d like to enhance your Notebook’s functionality (e.g. prettier tables, code folding), you can install **Jupyter extensions**:

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install
```

Once installed, you’ll see a new NBExtensions tab in your Notebook menu.

---

## Other Jupyter Products

Besides Jupyter Notebook, there are also:

* **JupyterLab** → a more modern interface
* **JupyterHub** → multi-user collaboration
* **Voila** → turn your Notebook into a web app

---

### 3. How Are Jupyter, GPUs, NVIDIA, and Large LLM Training Related?

* **GPUs (especially NVIDIA GPUs)** are accelerators for deep learning, dramatically speeding up model training.
* **Large LLMs** (like GPT, BERT, etc.) involve massive computations and require GPU clusters or even greater compute resources for efficient training.

The role of Jupyter:

* **Development and experimentation environment**: You can write training code, debug models, and run small-scale experiments in Jupyter. Its interactive nature makes it easy to quickly test model architectures, tune hyperparameters, and examine results.
* **Accessing GPUs**: While Jupyter itself doesn’t “provide” a GPU, if you’re running Jupyter on a machine with an NVIDIA GPU (and the right drivers and CUDA toolkit installed), your Python code (using frameworks like TensorFlow or PyTorch) can leverage the GPU for acceleration.
* **Remote training management**: Many teams run Jupyter Notebooks on remote servers equipped with GPUs, allowing them to operate training processes via a web interface instead of logging in directly through the command line.

---

### Summary:

| Role             | Description                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------- |
| Jupyter          | Interactive environment for writing code and experiments; great for development and debugging |
| GPU (NVIDIA)     | Hardware accelerator enabling fast computation for deep learning models                       |
| Large LLM Models | Large-scale deep learning models that require significant GPU power for efficient training    |

---
