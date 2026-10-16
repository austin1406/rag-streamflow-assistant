package com.austin.evalapi.service;

public class RunNotFoundException extends RuntimeException {

    public RunNotFoundException(long id) {
        super("No evaluation run with id " + id);
    }
}
