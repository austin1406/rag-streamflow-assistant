package com.austin.evalapi.web;

import com.austin.evalapi.dto.CreateRunRequest;
import com.austin.evalapi.dto.RunDetail;
import com.austin.evalapi.dto.RunSummary;
import com.austin.evalapi.service.EvalRunService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.net.URI;
import java.util.List;

@RestController
@RequestMapping("/api/runs")
public class EvalRunController {

    private final EvalRunService service;

    public EvalRunController(EvalRunService service) {
        this.service = service;
    }

    @PostMapping
    public ResponseEntity<RunDetail> create(@Valid @RequestBody CreateRunRequest request) {
        RunDetail created = service.create(request);
        return ResponseEntity.created(URI.create("/api/runs/" + created.summary().id())).body(created);
    }

    @GetMapping
    public List<RunSummary> list() {
        return service.list();
    }

    @GetMapping("/{id}")
    public RunDetail get(@PathVariable long id) {
        return service.get(id);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable long id) {
        service.delete(id);
        return ResponseEntity.noContent().build();
    }
}
