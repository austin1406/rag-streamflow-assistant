package com.austin.evalapi.web;

import com.austin.evalapi.dto.RunDetail;
import com.austin.evalapi.dto.RunSummary;
import com.austin.evalapi.service.EvalRunService;
import com.austin.evalapi.service.RunNotFoundException;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.time.Instant;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(EvalRunController.class)
class EvalRunControllerTest {

    private static final String VALID_BODY = """
            {"label": "baseline", "model": "llama3.2:1b", "topK": 5,
             "results": [{"question": "q1", "retrievalOk": true, "answerOk": null}]}
            """;

    @Autowired
    MockMvc mvc;

    @MockitoBean
    EvalRunService service;

    private static RunSummary summary(long id) {
        return new RunSummary(id, "baseline", "llama3.2:1b", 5, Instant.parse("2026-10-15T12:00:00Z"),
                1, 1, 1.0, 0, 0, null);
    }

    @Test
    void createReturns201WithLocationAndBody() throws Exception {
        when(service.create(any())).thenReturn(new RunDetail(summary(7), List.of()));

        mvc.perform(post("/api/runs").contentType(MediaType.APPLICATION_JSON).content(VALID_BODY))
                .andExpect(status().isCreated())
                .andExpect(header().string("Location", "/api/runs/7"))
                .andExpect(jsonPath("$.summary.id").value(7))
                .andExpect(jsonPath("$.summary.retrievalRate").value(1.0));
    }

    @Test
    void createRejectsInvalidInputWithFieldErrors() throws Exception {
        String invalid = """
                {"label": " ", "model": "m", "topK": 0, "results": []}
                """;

        mvc.perform(post("/api/runs").contentType(MediaType.APPLICATION_JSON).content(invalid))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors.label").exists())
                .andExpect(jsonPath("$.errors.topK").exists())
                .andExpect(jsonPath("$.errors.results").exists());
    }

    @Test
    void createRejectsABlankQuestionInsideAResult() throws Exception {
        String invalid = """
                {"label": "x", "model": "m", "topK": 5,
                 "results": [{"question": "", "retrievalOk": true}]}
                """;

        mvc.perform(post("/api/runs").contentType(MediaType.APPLICATION_JSON).content(invalid))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors['results[0].question']").exists());
    }

    @Test
    void createRejectsMalformedJson() throws Exception {
        mvc.perform(post("/api/runs").contentType(MediaType.APPLICATION_JSON).content("{not json"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void listReturnsSummaries() throws Exception {
        when(service.list()).thenReturn(List.of(summary(2), summary(1)));

        mvc.perform(get("/api/runs"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2))
                .andExpect(jsonPath("$[0].id").value(2));
    }

    @Test
    void getReturns404ForAMissingRun() throws Exception {
        when(service.get(99L)).thenThrow(new RunNotFoundException(99L));

        mvc.perform(get("/api/runs/99"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.detail").value("No evaluation run with id 99"));
    }

    @Test
    void deleteReturns204() throws Exception {
        mvc.perform(delete("/api/runs/4")).andExpect(status().isNoContent());

        verify(service).delete(4L);
    }

    @Test
    void deleteReturns404ForAMissingRun() throws Exception {
        doThrow(new RunNotFoundException(4L)).when(service).delete(4L);

        mvc.perform(delete("/api/runs/4")).andExpect(status().isNotFound());
    }
}
