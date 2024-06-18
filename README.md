### hello world!
Testing the first repo

---

### Q - should I put `.py` or only Jupyter notebooks (`.ipynb`) in Github?
##### by [Melonface](https://www.reddit.com/user/MelonFace/): 
> how I'd say you do it. I work in the industy.
> 1. Develop in jupyter / google colab.
>
> 2. Once it runs, find things or patterns you're doing many times, make functions for those in a `utils.py` or similar collection of tools you may reuse in the future.
>
> 3. Take any non-general functions in the notebook and move them to a `my_project_stuff.py` (obviously better name).
>     * `%load_ext autoreload native jupyter` plugin will let you change these functions without restarting the kernel.
>
> 5. Parameterize and move the model definition into `name_model.py`
> 6. Now your notebook should contain only the data loading and training code. This is fine. And still gives you the debug/experiment freedom.
>
> 7. Personally I also put the training code in a `.py` file at the end and load checkpoints into notebooks as needed to debug or fine tune. But that's personal taste. The reason for this is the service I use to run training in cloud. Where it's more reliable to run long running python programs rather than jupyter kernels.

##### by [ktpr](https://www.reddit.com/user/ktpr/):
> You could convert it to a [CLI integration](https://docs.github.com/en/github-cli/github-cli/quickstart) , a cooperative layout on Github -
> ```
>    pip install -U pipenv cookiecutter
>    cookiecutter gh:crmne/cookiecutter-modern-datascience
> ```
> Copy your `.ipynb` notebooks into `./notebook/`
> - Step 1: from your notebooks, place data, pipeline and model code into their respective directories. Wrap with Typer for CLI integration
> - Step 2: in `./serve`, make an `app.py` file that calls appropriate code to expose your ML project as an API. 
> - Endpoints can load, train, predict and otherwise serve your project.

---
### Q -  train-test splitting and evaluate performance metrics when creating a recommender system using collaborative filtering?

Answer by MIT expert:

> There are two main approaches to consider:
> 
> **Holdout**: This is the simplest method. Split your data into training and testing sets (common split is 80/20). Train your model on the training set and evaluate its performance on the unseen testing set.
> 
> **Cross-Validation**: This is a more robust approach. Divide your data into several folds (e.g., 10 folds). Train your model on all folds except one (validation fold), and evaluate on the validation fold. Repeat this process for each fold, averaging the results for a more reliable estimate.
> 
> #### Important Considerations:
> 
> **User-based** vs. **Item-based Splitting**: Decide whether to split by users or items. User-based splitting ensures users have interactions in both sets, but might not expose the model to unseen items. Item-based splitting can handle unseen items, but users might not have interactions in both sets.
> 
> **Cold Start Problem**: New users or items might not have enough data in the training set for accurate recommendations. Consider alternative approaches like item popularity or content-based filtering for these cases.
> 
> Here are some common metrics used to evaluate recommender systems:
> 
> **Root Mean Squared Error** (RMSE): Measures the average magnitude of the difference between predicted and actual ratings.
> 
> **Mean Absolute Error** (MAE): Similar to RMSE but uses absolute differences.
> 
> `Precision@k` and `Recall@k`: Measure how many relevant items are recommended among the top k recommendations.
> 
> **Normalized Discounted Cumulative Gain** (NDCG): Measures the ranking quality of recommendations, considering items with higher relevance being positioned higher.
> 
> #### Choosing the Right Metric:
> 
> The best metric depends on your specific goals. RMSE and MAE are good for predicting ratings, while `Precision@k` and `Recall@k` are better for capturing relevance. NDCG prioritizes highly relevant items.

---
### INFO - Python and Data Sciences YouTube Channels

1. [Data Analysis Using SPSS (49 videos)](https://lnkd.in/dE8e7H8R)
2. [Data Analysis Using Jamovi (26 videos)](https://lnkd.in/d4d_w9NR)
3. [Data Analysis Using Smart PLS-4 (35 videos)](https://lnkd.in/d-ZJ-Vkr)
4. [Time Series Analysis Using Eviews-12 (32 videos)](https://lnkd.in/dNthyy55)
5. [Panel Data Analysis Using Eviews-12 (25 videos)](https://lnkd.in/d2-KKKxV)
6. [Data Analysis Using Python (37 videos)](https://lnkd.in/dX8HDdxC)
7. [Data Analysis Using Rapidminer (70 videos)](https://lnkd.in/dpNTfA_X)
8. [Data Science Using R and R-Studio (28 videos)](https://lnkd.in/ddqQVHTg)
9. [Structural Equation Modeling using Smart PLS (22 videos)](https://lnkd.in/df2dzu7d)
10. [Qualitative Data Analysis in Nvivo (17 videos)](https://lnkd.in/deXZacwp)
11. [Machine Learning Using Python (15 videos)](https://lnkd.in/dDRMzzCT)
12. [Structural Equation Modelling using ADANCO (15 videos)](https://lnkd.in/d4EDmhdk)
13. [Data Visualization using Tableau (100 videos)](https://lnkd.in/dctQQjey)
14. [Data Analysis Using KNIME (139 videos)](https://lnkd.in/dp-GnY2C)
15. [Structural Equation Modeling using IBM SPSS AMOS](https://lnkd.in/d2dSKc7i)
16. [Artificial Intelligence (AI) Tools in Research](https://lnkd.in/dV5r56rW)
17. [Data Analysis Using Alteryx](https://lnkd.in/dHBaF-Zf)
18. [Data Analysis Using Excel](https://lnkd.in/dj9BbNjB)
19. [Data Analysis Using Power BI](https://lnkd.in/dPRQicV5)

---
### INFO - LLM / BUILDING / PROMPT / RAG / FINE-TUNING
#### LLM Basics and Foundations
1.  [Large Language Models](https://rycolab.io/classes/llm-s23/) by ETH
    Zurich
2.  [Understanding Large Language
    Models](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/) by
    Princeton
3.  [Transformers
    course](https://huggingface.co/learn/nlp-course/chapter1/1) by
    Huggingface
4.  [NLP course](https://huggingface.co/learn/nlp-course/chapter1/1) by
    Huggingface
5.  [CS324 - Large Language
    Models](https://stanford-cs324.github.io/winter2022/) by Stanford
6.  [Generative AI with Large Language
    Models](https://www.coursera.org/learn/generative-ai-with-llms) by
    Coursera
7.  [Introduction to Generative
    AI](https://www.coursera.org/learn/introduction-to-generative-ai) by
    Coursera
8.  [Generative AI
    Fundamentals](https://www.cloudskillsboost.google/paths/118/course_templates/556) by
    Google Cloud
9.  [Introduction to Large Language
    Models](https://www.cloudskillsboost.google/paths/118/course_templates/539) by
    Google Cloud
10. [Introduction to Generative
    AI](https://www.cloudskillsboost.google/paths/118/course_templates/536) by
    Google Cloud
11. [Generative AI
    Concepts](https://www.datacamp.com/courses/generative-ai-concepts) by
    DataCamp (Daniel Tedesco Data Lead @ Google)
12. [1 Hour Introduction to LLM (Large Language
    Models)](https://www.youtube.com/watch?v=xu5_kka-suc) by WeCloudData
13. [LLM Foundation Models from the Ground Up \|
    Primer](https://www.youtube.com/watch?v=W0c7jQezTDw&list=PLTPXxbhUt-YWjMCDahwdVye8HW69p5NYS) by
    Databricks
14. [Generative AI
    Explained](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-07+V1/) by
    Nvidia
15. [Transformer Models and BERT
    Model](https://www.cloudskillsboost.google/course_templates/538) by
    Google Cloud
16. [Generative AI Learning Plan for Decision
    Makers](https://explore.skillbuilder.aws/learn/public/learning_plan/view/1909/generative-ai-learning-plan-for-decision-makers) by
    AWS
17. [Introduction to Responsible
    AI](https://www.cloudskillsboost.google/course_templates/554) by
    Google Cloud
18. [Fundamentals of Generative
    AI](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/) by
    Microsoft Azure
19. [Generative AI for
    Beginners](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-122979-leestott) by
    Microsoft
20. [ChatGPT for Beginners: The Ultimate Use Cases for
    Everyone](https://www.udemy.com/course/chatgpt-for-beginners-the-ultimate-use-cases-for-everyone/) by
    Udemy
21. [\[1hr Talk\] Intro to Large Language
    Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) by Andrej
    Karpathy
22. [ChatGPT for
    Everyone](https://learnprompting.org/courses/chatgpt-for-everyone) by
    Learn Prompting
23. [Large Language Models (LLMs) (In
    English)](https://www.youtube.com/playlist?list=PLxlkzujLkmQ9vMaqfvqyfvZV_o8EqjAk7) by
    Kshitiz Verma (JK Lakshmipat University, Jaipur, India)
#### Building LLM Applications
1.  [LLMOps: Building Real-World Applications With Large Language
    Models](https://www.udacity.com/course/building-real-world-applications-with-large-language-models--cd13455) by
    Udacity
2.  [Full Stack LLM
    Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) by FSDL
3.  [Generative AI for
    beginners](https://github.com/microsoft/generative-ai-for-beginners/tree/main) by
    Microsoft
4.  [Large Language Models: Application through
    Production](https://www.edx.org/learn/computer-science/databricks-large-language-models-application-through-production) by
    Databricks
5.  [Generative AI
    Foundations](https://www.youtube.com/watch?v=oYm66fHqHUM&list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF) by
    AWS
6.  [Introduction to Generative AI Community
    Course](https://www.youtube.com/watch?v=ajWheP8ZD70&list=PLmQAMKHKeLZ-iTT-E2kK9uePrJ1Xua9VL) by
    ineuron
7.  [LLM University](https://docs.cohere.com/docs/llmu) by Cohere
8.  [LLM Learning Lab](https://lightning.ai/pages/llm-learning-lab/) by
    Lightning AI
9.  [Functions, Tools and Agents with
    LangChain](https://learn.deeplearning.ai/functions-tools-agents-langchain) by
    Deeplearning.AI
10. [LangChain for LLM Application
    Development](https://learn.deeplearning.ai/login?redirect_course=langchain&callbackUrl=https%3A%2F%2Flearn.deeplearning.ai%2Fcourses%2Flangchain) by
    Deeplearning.AI
11. [LLMOps](https://learn.deeplearning.ai/llmops) by DeepLearning.AI
12. [Automated Testing for
    LLMOps](https://learn.deeplearning.ai/automated-testing-llmops) by
    DeepLearning.AI
13. [Building RAG Agents with
    LLMs](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-15+V1/) by
    Nvidia
14. [Building Generative AI Applications Using Amazon
    Bedrock](https://explore.skillbuilder.aws/learn/course/external/view/elearning/17904/building-generative-ai-applications-using-amazon-bedrock-aws-digital-training) by
    AWS
15. [Efficiently Serving
    LLMs](https://learn.deeplearning.ai/courses/efficiently-serving-llms/lesson/1/introduction) by
    DeepLearning.AI
16. [Building Systems with the ChatGPT
    API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) by
    DeepLearning.AI
17. [Serverless LLM apps with Amazon
    Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/) by
    DeepLearning.AI
18. [Building Applications with Vector
    Databases](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) by
    DeepLearning.AI
19. [Automated Testing for
    LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) by
    DeepLearning.AI
20. [LLMOps](https://www.deeplearning.ai/short-courses/llmops/) by
    DeepLearning.AI
21. [Build LLM Apps with
    LangChain.js](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/) by
    DeepLearning.AI
22. [Advanced Retrieval for AI with
    Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) by
    DeepLearning.AI
23. [Operationalizing LLMs on
    Azure](https://www.coursera.org/learn/llmops-azure) by Coursera
24. [Generative AI Full Course -- Gemini Pro, OpenAI, Llama, Langchain,
    Pinecone, Vector Databases &
    More](https://www.youtube.com/watch?v=mEsleV16qdo) by
    freeCodeCamp.org
25. [Training & Fine-Tuning LLMs for
    Production](https://learn.activeloop.ai/courses/llms) by Activeloop
#### Prompt Engineering, RAG and Fine-Tuning
1.  [LangChain & Vector Databases in
    Production](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVhnQW8xNDdhSU9IUDVLXzFhV2N0UkNRMkZrQXxBQ3Jtc0traUxHMzZJcGJQYjlyckYxaGxYVWlsOFNGUFlFVEdhNzdjTWpPUlQ2TF9XczRqNkxMVGpJTnd5YmYzV0prQ0IwZURNcHhIZ3h1Z051VTl5MXBBLUN0dkM0NHRkQTFua1Jpc0VCRFJUb0ZQZG95b0JqMA&q=https%3A%2F%2Flearn.activeloop.ai%2Fcourses%2Flangchain&v=gKUTDC13jys) by
    Activeloop
2.  [Reinforcement Learning from Human
    Feedback](https://learn.deeplearning.ai/reinforcement-learning-from-human-feedback) by
    DeepLearning.AI
3.  [Building Applications with Vector
    Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by
    DeepLearning.AI
4.  [Finetuning Large Language
    Models](https://learn.deeplearning.ai/finetuning-large-language-models) by
    Deeplearning.AI
5.  [LangChain: Chat with Your
    Data](http://learn.deeplearning.ai/langchain-chat-with-your-data/) by
    Deeplearning.AI
6.  [Building Systems with the ChatGPT
    API](https://learn.deeplearning.ai/chatgpt-building-system) by
    Deeplearning.AI
7.  [Prompt Engineering with Llama
    2](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/) by
    Deeplearning.AI
8.  [Building Applications with Vector
    Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by
    Deeplearning.AI
9.  [ChatGPT Prompt Engineering for
    Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction) by
    Deeplearning.AI
10. [Advanced RAG Orchestration
    series](https://www.youtube.com/watch?v=CeDS1yvw9E4) by LlamaIndex
11. [Prompt Engineering
    Specialization](https://www.coursera.org/specializations/prompt-engineering) by
    Coursera
12. [Augment your LLM Using Retrieval Augmented
    Generation](https://courses.nvidia.com/courses/course-v1:NVIDIA+S-FX-16+v1/) by
    Nvidia
13. [Knowledge Graphs for
    RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) by
    Deeplearning.AI
14. [Open Source Models with Hugging
    Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/) by
    Deeplearning.AI
15. [Vector Databases: from Embeddings to
    Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) by
    Deeplearning.AI
16. [Understanding and Applying Text
    Embeddings](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/) by
    Deeplearning.AI
17. [JavaScript RAG Web Apps with
    LlamaIndex](https://www.deeplearning.ai/short-courses/javascript-rag-web-apps-with-llamaindex/) by
    Deeplearning.AI
18. [Quantization Fundamentals with Hugging
    Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) by
    Deeplearning.AI
19. [Preprocessing Unstructured Data for LLM
    Applications](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/) by
    Deeplearning.AI
20. [Retrieval Augmented Generation for Production with LangChain &
    LlamaIndex](https://learn.activeloop.ai/courses/rag) by Activeloop
21. [Quantization in
    Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/) by
    Deeplearning.AI

---
### Text operations Tutorial
###### by U of Penn. Library:
> https://guides.library.upenn.edu/penntdm/API

---
> ###### Prepared by:
> ###### Sun CHUNG, *SMIEEE* M.Sc. HKU

---
###### End of File.


