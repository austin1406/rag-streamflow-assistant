package com.austin.evalapi.repo;

import com.austin.evalapi.model.EvalRun;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface EvalRunRepository extends JpaRepository<EvalRun, Long> {

    @EntityGraph(attributePaths = "results")
    List<EvalRun> findAllByOrderByCreatedAtDescIdDesc();

    @EntityGraph(attributePaths = "results")
    Optional<EvalRun> findWithResultsById(Long id);
}
