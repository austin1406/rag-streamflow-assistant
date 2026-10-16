package com.austin.evalapi.service;

import com.austin.evalapi.dto.CreateRunRequest;
import com.austin.evalapi.dto.CreateRunRequest.ResultInput;
import com.austin.evalapi.dto.RunDetail;
import com.austin.evalapi.dto.RunSummary;
import com.austin.evalapi.model.EvalResult;
import com.austin.evalapi.model.EvalRun;
import com.austin.evalapi.repo.EvalRunRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EvalRunServiceTest {

    private static final Instant NOW = Instant.parse("2026-10-15T12:00:00Z");

    @Mock
    EvalRunRepository repository;

    EvalRunService service;

    @BeforeEach
    void setUp() {
        service = new EvalRunService(repository, Clock.fixed(NOW, ZoneOffset.UTC));
    }

    @Test
    void createTrimsInputStampsTheTimeAndSavesAllResults() {
        when(repository.save(any(EvalRun.class))).thenAnswer(inv -> inv.getArgument(0));

        RunDetail detail = service.create(new CreateRunRequest(
                "  baseline  ", " llama3.2:1b ", 5,
                List.of(new ResultInput(" q1 ", true, true), new ResultInput("q2", false, null))));

        ArgumentCaptor<EvalRun> saved = ArgumentCaptor.forClass(EvalRun.class);
        verify(repository).save(saved.capture());
        EvalRun run = saved.getValue();
        assertThat(run.getLabel()).isEqualTo("baseline");
        assertThat(run.getModel()).isEqualTo("llama3.2:1b");
        assertThat(run.getCreatedAt()).isEqualTo(NOW);
        assertThat(run.getResults()).extracting(EvalResult::getQuestion).containsExactly("q1", "q2");

        assertThat(detail.summary().questions()).isEqualTo(2);
        assertThat(detail.summary().retrievalHits()).isEqualTo(1);
        assertThat(detail.results()).hasSize(2);
    }

    @Test
    void listReturnsSummariesInRepositoryOrder() {
        EvalRun newer = new EvalRun("newer", "m", 5, NOW);
        newer.addResult(new EvalResult("q", true, null));
        EvalRun older = new EvalRun("older", "m", 5, NOW.minusSeconds(60));
        older.addResult(new EvalResult("q", false, null));
        when(repository.findAllByOrderByCreatedAtDescIdDesc()).thenReturn(List.of(newer, older));

        List<RunSummary> summaries = service.list();

        assertThat(summaries).extracting(RunSummary::label).containsExactly("newer", "older");
        assertThat(summaries).extracting(RunSummary::retrievalRate).containsExactly(1.0, 0.0);
    }

    @Test
    void getReturnsTheRunWhenItExists() {
        EvalRun run = new EvalRun("baseline", "m", 5, NOW);
        run.addResult(new EvalResult("q", true, true));
        when(repository.findWithResultsById(7L)).thenReturn(Optional.of(run));

        assertThat(service.get(7L).summary().label()).isEqualTo("baseline");
    }

    @Test
    void getThrowsWhenTheRunIsMissing() {
        when(repository.findWithResultsById(99L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.get(99L))
                .isInstanceOf(RunNotFoundException.class)
                .hasMessageContaining("99");
    }

    @Test
    void deleteRemovesAnExistingRun() {
        when(repository.existsById(3L)).thenReturn(true);

        service.delete(3L);

        verify(repository).deleteById(3L);
    }

    @Test
    void deleteThrowsAndDeletesNothingWhenTheRunIsMissing() {
        when(repository.existsById(3L)).thenReturn(false);

        assertThatThrownBy(() -> service.delete(3L)).isInstanceOf(RunNotFoundException.class);
        verify(repository, never()).deleteById(any());
    }
}
