package com.intentmedia.admm;

import com.google.common.base.Optional;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class AdmmOptimizerDriver extends Configured implements Tool {

    private static final int DEFAULT_ADMM_ITERATIONS_MAX = 5; // max number of iteration
    private static final float DEFAULT_REGULARIZATION_FACTOR = 0.000001f;
    private static final String S3_ITERATION_FOLDER_NAME = "iteration_"; // output folder for each iteration

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new Configuration(), new AdmmOptimizerDriver(), args); 
    }

    @Override
    public int run(String[] args) throws IOException, CmdLineException {
        AdmmOptimizerDriverArguments admmOptimizerDriverArguments = new AdmmOptimizerDriverArguments();
        parseArgs(args, admmOptimizerDriverArguments);

        String signalDataLocation = admmOptimizerDriverArguments.getSignalPath(); // get inputPath from terminal
        String intermediateHdfsBaseString = "/tmp";
        String finalOutputBaseUrl = admmOptimizerDriverArguments.getOutputPath(); // get outputPath 
        int iterationsMaximum = Optional.fromNullable(admmOptimizerDriverArguments.getIterationsMaximum()).or(
                DEFAULT_ADMM_ITERATIONS_MAX);
        float regularizationFactor = Optional.fromNullable(admmOptimizerDriverArguments.getRegularizationFactor()).or(
                DEFAULT_REGULARIZATION_FACTOR);
        boolean addIntercept = Optional.fromNullable(admmOptimizerDriverArguments.getAddIntercept()).or(false);
        boolean regularizeIntercept = Optional.fromNullable(admmOptimizerDriverArguments.getRegularizeIntercept()).or(false);
        String columnsToExclude = Optional.fromNullable(admmOptimizerDriverArguments.getColumnsToExclude()).or("");

        int iterationNumber = 0;
        boolean isFinalIteration = false;

        while (!isFinalIteration) {
            long preStatus = 0;
            JobConf conf = new JobConf(getConf(), AdmmOptimizerDriver.class); // configure a job
            Path previousHdfsResultsPath = new Path(finalOutputBaseUrl + intermediateHdfsBaseString + S3_ITERATION_FOLDER_NAME + (iterationNumber - 1)); // path of previous iteation
            Path currentHdfsResultsPath = new Path(finalOutputBaseUrl + intermediateHdfsBaseString + S3_ITERATION_FOLDER_NAME + iterationNumber); // path of current iteration

            // run map reduce job here
            long curStatus = doAdmmIteration(conf,
                    previousHdfsResultsPath,
                    currentHdfsResultsPath,
                    signalDataLocation,
                    iterationNumber,
                    columnsToExclude,
                    addIntercept,
                    regularizeIntercept,
                    regularizationFactor); // run a mapreduce job
            isFinalIteration = convergedOrMaxed(curStatus, preStatus, iterationNumber, iterationsMaximum);

            iterationNumber++;
        }

        return 0;
    }

    private void parseArgs(String[] args, AdmmOptimizerDriverArguments admmOptimizerDriverArguments) throws CmdLineException {
        ArrayList<String> argsList = new ArrayList<String>(Arrays.asList(args));

        for (int i = 0; i < args.length; i++) {
            if (i % 2 == 0 && !AdmmOptimizerDriverArguments.VALID_ARGUMENTS.contains(args[i])) {
                argsList.remove(args[i]);
                argsList.remove(args[i + 1]);
            }
        }

        new CmdLineParser(admmOptimizerDriverArguments).parseArgument(argsList.toArray(new String[argsList.size()]));
    }


 

    
// function to run MapReduce job
    public long doAdmmIteration(JobConf conf,
                                Path previousHdfsPath,
                                Path currentHdfsPath,
                                String signalDataLocation,
                                int iterationNumber,
                                String columnsToExclude,
                                boolean addIntercept,
                                boolean regularizeIntercept,
                                float regularizationFactor) throws IOException {
        Path signalDataInputLocation = new Path(signalDataLocation);

        conf.setJobName("ADMM Optimizer " + iterationNumber);
        conf.set("mapred.child.java.opts", "-Xmx2g");
        conf.set("previous.intermediate.output.location", previousHdfsPath.toString());
        conf.setInt("iteration.number", iterationNumber);
        conf.set("columns.to.exclude", columnsToExclude);
        conf.setBoolean("add.intercept", addIntercept);
        conf.setBoolean("regularize.intercept", regularizeIntercept);
        conf.setFloat("regularization.factor", regularizationFactor);
        conf.setFloat("mapred.reduce.slowstart.completed.maps", 1.0f);

        conf.setMapperClass(AdmmIterationMapper.class);
        conf.setReducerClass(AdmmIterationReducer.class);
        conf.setMapOutputKeyClass(IntWritable.class);
        conf.setMapOutputValueClass(Text.class);
        conf.setOutputKeyClass(IntWritable.class);
        conf.setOutputValueClass(Text.class);
        conf.setInputFormat(TextInputFormat.class);
        conf.setOutputFormat(TextOutputFormat.class);

        FileInputFormat.setInputPaths(conf, signalDataInputLocation);
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(currentHdfsPath)) {
            fs.delete(currentHdfsPath, true);
        }
        FileOutputFormat.setOutputPath(conf, currentHdfsPath);

        RunningJob job = JobClient.runJob(conf);

        return job.getCounters().findCounter(AdmmIterationReducer.IterationCounter.ITERATION).getValue();
    }

    private boolean convergedOrMaxed(long curStatus, long preStatus, int iterationNumber, int iterationsMaximum) {
        return iterationNumber >= DEFAULT_ADMM_ITERATIONS_MAX;
   
    }
}
