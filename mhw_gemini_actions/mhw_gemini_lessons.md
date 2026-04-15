# MHW Gemini Lessons Learned

- [2026-04-10] **Regression Task vs. Classification:** Clarified that the project is a regression task (predicting continuous SDD values). There are no confusion matrices, Precision, Recall, or F1 metrics. Model validation is performed by monitoring **Mean Squared Error (MSE)** and **Loss Curves** (Train vs. Val) to detect over- or under-fitting.
- [2026-04-14] **Workflow - Task Movement:** Completed tasks in `mhw_gemini_todo.md` MUST be moved to `mhw_gemini_recentactions.md` with an updated timestamp and summary of findings.
