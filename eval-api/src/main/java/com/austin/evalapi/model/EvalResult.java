package com.austin.evalapi.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "eval_result")
public class EvalResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "run_id")
    private EvalRun run;

    @Column(nullable = false, length = 500)
    private String question;

    @Column(nullable = false)
    private boolean retrievalOk;

    /** Null when no generation backend was available to grade the answer. */
    private Boolean answerOk;

    protected EvalResult() {
    }

    public EvalResult(String question, boolean retrievalOk, Boolean answerOk) {
        this.question = question;
        this.retrievalOk = retrievalOk;
        this.answerOk = answerOk;
    }

    void setRun(EvalRun run) {
        this.run = run;
    }

    public Long getId() {
        return id;
    }

    public String getQuestion() {
        return question;
    }

    public boolean isRetrievalOk() {
        return retrievalOk;
    }

    public Boolean getAnswerOk() {
        return answerOk;
    }
}
