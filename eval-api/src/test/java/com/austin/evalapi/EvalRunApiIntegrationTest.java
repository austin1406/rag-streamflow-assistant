package com.austin.evalapi;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class EvalRunApiIntegrationTest {

    private static final String BODY = """
            {"label": "integration", "model": "llama3.2:1b", "topK": 5, "results": [
              {"question": "q1", "retrievalOk": true,  "answerOk": true},
              {"question": "q2", "retrievalOk": true,  "answerOk": false},
              {"question": "q3", "retrievalOk": false, "answerOk": false},
              {"question": "q4", "retrievalOk": true,  "answerOk": null}
            ]}
            """;

    @Autowired
    MockMvc mvc;

    @Test
    void createReadListAndDeleteARunThroughTheWholeStack() throws Exception {
        MvcResult created = mvc.perform(post("/api/runs").contentType(MediaType.APPLICATION_JSON).content(BODY))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.summary.questions").value(4))
                .andExpect(jsonPath("$.summary.retrievalHits").value(3))
                .andExpect(jsonPath("$.summary.retrievalRate").value(0.75))
                .andExpect(jsonPath("$.summary.answersGraded").value(3))
                .andExpect(jsonPath("$.summary.answerHits").value(1))
                .andReturn();

        String location = created.getResponse().getHeader("Location");
        assertThat(location).startsWith("/api/runs/");

        mvc.perform(get(location))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results.length()").value(4))
                .andExpect(jsonPath("$.results[0].question").value("q1"));

        mvc.perform(get("/api/runs"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[?(@.label == 'integration')]").exists());

        mvc.perform(delete(location)).andExpect(status().isNoContent());
        mvc.perform(get(location)).andExpect(status().isNotFound());
    }

    @Test
    void invalidRequestsAreRejectedBeforeTouchingTheDatabase() throws Exception {
        mvc.perform(post("/api/runs").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"label\": \"x\", \"model\": \"m\", \"topK\": 99, \"results\": []}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors.topK").exists());
    }
}
