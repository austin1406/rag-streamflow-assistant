package com.austin.evalapi.repo;

import com.austin.evalapi.model.EvalResult;
import com.austin.evalapi.model.EvalRun;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.data.jpa.test.autoconfigure.DataJpaTest;

import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest
class EvalRunRepositoryTest {

    @Autowired
    EvalRunRepository repository;

    private EvalRun saveRun(String label, Instant createdAt, String... questions) {
        EvalRun run = new EvalRun(label, "model", 5, createdAt);
        for (String question : questions) {
            run.addResult(new EvalResult(question, true, null));
        }
        return repository.saveAndFlush(run);
    }

    @Test
    void savesAndLoadsResultsWithTheRun() {
        EvalRun saved = saveRun("baseline", Instant.parse("2026-10-15T12:00:00Z"), "q1", "q2", "q3");

        EvalRun loaded = repository.findWithResultsById(saved.getId()).orElseThrow();

        assertThat(loaded.getResults()).extracting(EvalResult::getQuestion).containsExactly("q1", "q2", "q3");
    }

    @Test
    void returnsEmptyForAnUnknownId() {
        assertThat(repository.findWithResultsById(12345L)).isEmpty();
    }

    @Test
    void listsNewestRunsFirst() {
        saveRun("oldest", Instant.parse("2026-10-01T00:00:00Z"), "q");
        saveRun("newest", Instant.parse("2026-10-20T00:00:00Z"), "q");
        saveRun("middle", Instant.parse("2026-10-10T00:00:00Z"), "q");

        List<EvalRun> runs = repository.findAllByOrderByCreatedAtDescIdDesc();

        assertThat(runs).extracting(EvalRun::getLabel).containsExactly("newest", "middle", "oldest");
    }

    @Test
    void deletingARunDeletesItsResults() {
        EvalRun saved = saveRun("baseline", Instant.parse("2026-10-15T12:00:00Z"), "q1", "q2");

        repository.deleteById(saved.getId());
        repository.flush();

        assertThat(repository.existsById(saved.getId())).isFalse();
    }
}
