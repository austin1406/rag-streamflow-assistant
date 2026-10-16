package com.austin.evalapi.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;

import java.util.List;

public record CreateRunRequest(
        @NotBlank @Size(max = 100) String label,
        @NotBlank @Size(max = 100) String model,
        @Min(1) @Max(20) int topK,
        @NotEmpty @Size(max = 500) List<@Valid ResultInput> results) {

    public record ResultInput(
            @NotBlank @Size(max = 500) String question,
            boolean retrievalOk,
            Boolean answerOk) {
    }
}
