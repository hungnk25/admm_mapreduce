package com.intentmedia.admm;

import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.spi.BooleanOptionHandler;
import org.kohsuke.args4j.spi.FloatOptionHandler;
import org.kohsuke.args4j.spi.IntOptionHandler;
import org.kohsuke.args4j.spi.StringOptionHandler;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class AdmmOptimizerDriverArguments {

    public static final Set<String> VALID_ARGUMENTS = new HashSet<String>(Arrays.asList(
            "-outputPath",
            "-signalPath",
            "-iterationsMaximum",
            "-regularizationFactor",
            "-addIntercept",
            "-regularizeIntercept",
            "-columnsToExclude"));

    @Option(name = "-outputPath", required = true, handler = StringOptionHandler.class)
    private String outputPath;

    @Option(name = "-signalPath", required = true, handler = StringOptionHandler.class)
    private String signalPath;

    @Option(name = "-iterationsMaximum", required = false, handler = IntOptionHandler.class)
    private int iterationsMaximum;

    @Option(name = "-regularizationFactor", required = false, handler = FloatOptionHandler.class)
    private float regularizationFactor;

    @Option(name = "-addIntercept", required = false, handler = BooleanOptionHandler.class)
    private boolean addIntercept;

    @Option(name = "-regularizeIntercept", required = false, handler = BooleanOptionHandler.class)
    private boolean regularizeIntercept;

    @Option(name = "-columnsToExclude", required = false, handler = StringOptionHandler.class)
    private String columnsToExclude;

    public String getOutputPath() {
        return outputPath;
    }

    public String getSignalPath() {
        return signalPath;
    }

    public int getIterationsMaximum() {
        return iterationsMaximum;
    }

    public float getRegularizationFactor() {
        return regularizationFactor;
    }

    public boolean getAddIntercept() {
        return addIntercept;
    }

    public boolean getRegularizeIntercept() {
        return regularizeIntercept;
    }

    public String getColumnsToExclude() {
        return columnsToExclude;
    }
}
