package com.austin.evalapi.dto;

import com.austin.evalapi.model.EvalResult;
import com.austin.evalapi.model.EvalRun;

import java.time.Instant;
import java.util.List;

/**
 * Hit rates for one evaluation run. {@code answerRate} is null when no result in the run had a
 * graded answer, because dividing by zero would report a misleading 0%.
 */
public record RunSummary(
        Long id,
        String label,
        String model,
        int topK,
        Instant createdAt,
        int questions,
        int retrievalHits,
        double retrievalRate,
        int answersGraded,
        int answerHits,
        Double answerRate) {

    public static RunSummary from(EvalRun run) {
        List<EvalResult> results = run.getResults();
        int questions = results.size();
        int retrievalHits = (int) results.stream().filter(EvalResult::isRetrievalOk).count();
        int graded = (int) results.stream().filter(r -> r.getAnswerOk() != null).count();
        int answerHits = (int) results.stream().filter(r -> Boolean.TRUE.equals(r.getAnswerOk())).count();

        return new RunSummary(
                run.getId(),
                run.getLabel(),
                run.getModel(),
                run.getTopK(),
                run.getCreatedAt(),
                questions,
                retrievalHits,
                rate(retrievalHits, questions),
                graded,
                answerHits,
                graded == 0 ? null : rate(answerHits, graded));
    }

    private static double rate(int hits, int total) {
        return total == 0 ? 0.0 : Math.round(10000.0 * hits / total) / 10000.0;
    }
}
