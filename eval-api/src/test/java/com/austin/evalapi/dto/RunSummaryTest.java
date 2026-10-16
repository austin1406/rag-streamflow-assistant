package com.austin.evalapi.dto;

import com.austin.evalapi.model.EvalResult;
import com.austin.evalapi.model.EvalRun;
import org.junit.jupiter.api.Test;

import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

class RunSummaryTest {

    private static EvalRun run() {
        return new EvalRun("baseline", "llama3.2:1b", 5, Instant.parse("2026-10-15T12:00:00Z"));
    }

    @Test
    void computesRetrievalAndAnswerRates() {
        EvalRun run = run();
        run.addResult(new EvalResult("q1", true, true));
        run.addResult(new EvalResult("q2", true, false));
        run.addResult(new EvalResult("q3", false, false));
        run.addResult(new EvalResult("q4", true, true));

        RunSummary summary = RunSummary.from(run);

        assertThat(summary.questions()).isEqualTo(4);
        assertThat(summary.retrievalHits()).isEqualTo(3);
        assertThat(summary.retrievalRate()).isEqualTo(0.75);
        assertThat(summary.answersGraded()).isEqualTo(4);
        assertThat(summary.answerHits()).isEqualTo(2);
        assertThat(summary.answerRate()).isEqualTo(0.5);
    }

    @Test
    void answerRateIsNullWhenNothingWasGraded() {
        EvalRun run = run();
        run.addResult(new EvalResult("q1", true, null));
        run.addResult(new EvalResult("q2", false, null));

        RunSummary summary = RunSummary.from(run);

        assertThat(summary.answersGraded()).isZero();
        assertThat(summary.answerHits()).isZero();
        assertThat(summary.answerRate()).isNull();
        assertThat(summary.retrievalRate()).isEqualTo(0.5);
    }

    @Test
    void onlyGradedAnswersCountTowardTheAnswerRate() {
        EvalRun run = run();
        run.addResult(new EvalResult("q1", true, true));
        run.addResult(new EvalResult("q2", true, null));
        run.addResult(new EvalResult("q3", true, null));

        RunSummary summary = RunSummary.from(run);

        assertThat(summary.answersGraded()).isEqualTo(1);
        assertThat(summary.answerRate()).isEqualTo(1.0);
    }

    @Test
    void roundsRatesToFourDecimals() {
        EvalRun run = run();
        run.addResult(new EvalResult("q1", true, null));
        run.addResult(new EvalResult("q2", false, null));
        run.addResult(new EvalResult("q3", false, null));

        assertThat(RunSummary.from(run).retrievalRate()).isEqualTo(0.3333);
    }

    @Test
    void emptyRunHasZeroRates() {
        RunSummary summary = RunSummary.from(run());

        assertThat(summary.questions()).isZero();
        assertThat(summary.retrievalRate()).isZero();
        assertThat(summary.answerRate()).isNull();
    }
}
