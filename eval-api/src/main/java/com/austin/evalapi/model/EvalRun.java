package com.austin.evalapi.model;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OrderBy;
import jakarta.persistence.Table;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "eval_run")
public class EvalRun {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 100)
    private String label;

    @Column(nullable = false, length = 100)
    private String model;

    @Column(nullable = false)
    private int topK;

    @Column(nullable = false)
    private Instant createdAt;

    @OneToMany(mappedBy = "run", cascade = CascadeType.ALL, orphanRemoval = true)
    @OrderBy("id")
    private List<EvalResult> results = new ArrayList<>();

    protected EvalRun() {
    }

    public EvalRun(String label, String model, int topK, Instant createdAt) {
        this.label = label;
        this.model = model;
        this.topK = topK;
        this.createdAt = createdAt;
    }

    public void addResult(EvalResult result) {
        result.setRun(this);
        results.add(result);
    }

    public Long getId() {
        return id;
    }

    public String getLabel() {
        return label;
    }

    public String getModel() {
        return model;
    }

    public int getTopK() {
        return topK;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public List<EvalResult> getResults() {
        return results;
    }
}
