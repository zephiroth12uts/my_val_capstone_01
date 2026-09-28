# myVal Next-Category Recommendation

Machine learning prototype developed for the myVal Capstone Project.

The objective of this project is to recommend the **next household-content category** that may be relevant for a customer to document, using the information already available within myVal.

The recommendation is sequential: as the customer adds more assets and the household state evolves, the model can generate a new recommendation.

---

## Project Objective

The modelling task is defined as:

> Given the information already documented and analysed for a household, predict the next previously undocumented category that may be relevant for the customer to capture.

The model combines:

- documented household assets;
- existing myVal AI-analysis outputs;
- customer and household information;
- property context;
- optional property-video information.

The system is designed as a recommendation tool, not as a claim that a category is definitely missing.

---

## Main Workflow

```text
Customer documents an asset
        ↓
myVal AI analyses the asset
        ↓
Household state is updated
        ↓
Next-category recommendation model
        ↓
Top-1 / Top-2 / Top-3 recommendations