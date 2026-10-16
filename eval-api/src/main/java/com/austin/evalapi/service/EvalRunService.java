package com.austin.evalapi.service;

import com.austin.evalapi.dto.CreateRunRequest;
import com.austin.evalapi.dto.RunDetail;
import com.austin.evalapi.dto.RunSummary;
import com.austin.evalapi.model.EvalResult;
import com.austin.evalapi.model.EvalRun;
import com.austin.evalapi.repo.EvalRunRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Clock;
import java.time.Instant;
import java.util.List;

@Service
public class EvalRunService {

    private final EvalRunRepository repository;
    private final Clock clock;

    public EvalRunService(EvalRunRepository repository, Clock clock) {
        this.repository = repository;
        this.clock = clock;
    }

    @Transactional
    public RunDetail create(CreateRunRequest request) {
        EvalRun run = new EvalRun(request.label().trim(), request.model().trim(), request.topK(), Instant.now(clock));
        for (CreateRunRequest.ResultInput input : request.results()) {
            run.addResult(new EvalResult(input.question().trim(), input.retrievalOk(), input.answerOk()));
        }
        return RunDetail.from(repository.save(run));
    }

    @Transactional(readOnly = true)
    public List<RunSummary> list() {
        return repository.findAllByOrderByCreatedAtDescIdDesc().stream().map(RunSummary::from).toList();
    }

    @Transactional(readOnly = true)
    public RunDetail get(long id) {
        return repository.findWithResultsById(id).map(RunDetail::from).orElseThrow(() -> new RunNotFoundException(id));
    }

    @Transactional
    public void delete(long id) {
        if (!repository.existsById(id)) {
            throw new RunNotFoundException(id);
        }
        repository.deleteById(id);
    }
}
