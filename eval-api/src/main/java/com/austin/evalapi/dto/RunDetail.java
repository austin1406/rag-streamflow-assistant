package com.austin.evalapi.dto;

import com.austin.evalapi.model.EvalResult;
import com.austin.evalapi.model.EvalRun;

import java.util.List;

public record RunDetail(RunSummary summary, List<ResultView> results) {

    public record ResultView(Long id, String question, boolean retrievalOk, Boolean answerOk) {

        static ResultView from(EvalResult result) {
            return new ResultView(
                    result.getId(), result.getQuestion(), result.isRetrievalOk(), result.getAnswerOk());
        }
    }

    public static RunDetail from(EvalRun run) {
        return new RunDetail(
                RunSummary.from(run),
                run.getResults().stream().map(ResultView::from).toList());
    }
}
