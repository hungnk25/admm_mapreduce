package com.intentmedia.admm;

import org.apache.hadoop.io.Text;

import java.io.IOException;
import java.util.Iterator;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import static com.intentmedia.admm.AdmmIterationHelper.jsonToAdmmReducerContext;

public class AdmmReducerContextGroup {

    private static final Pattern KEY_VALUE_DELIMITER = Pattern.compile("::");
    private final double[][] LambdaGroup; // collect all Lambda from Mappers
    private final double[][][] MuGroup; // collect all Mu from Mapper
    private final double[][] Y0kGroup;
    private final double[][][] X0kGroup;
    private final String[] splitIds;
    private final int numberOfMappers;


    public AdmmReducerContextGroup(Iterator<Text> mapperResults, int numberOfMappers, Logger logger, int iteration)
            throws IOException {
        this.numberOfMappers = numberOfMappers;
        String[] result = getNextResult(mapperResults);
        AdmmReducerContext context = jsonToAdmmReducerContext(result[1]);
        String splitId = result[0];
        logger.info(String.format("Iteration %d Reducer Getting splitId %s", iteration, splitId));


        LambdaGroup = new double[numberOfMappers][]; // each row contains Lambda from a mapper
        MuGroup = new double[numberOfMappers][][];
        Y0kGroup = new double[numberOfMappers][];
        X0kGroup = new double[numberOfMappers][][];
        
        splitIds = new String[numberOfMappers];

        int contextNumber = 0;

        while (result != null) {
            splitId = result[0];
            context = jsonToAdmmReducerContext(result[1]);
            LambdaGroup[contextNumber] = context.getLambda();
            MuGroup[contextNumber] = context.getMu();
            Y0kGroup[contextNumber] = context.getY0k();
            X0kGroup[contextNumber] = context.getX0k();

            splitIds[contextNumber] = splitId;
            result = getNextResult(mapperResults);
            contextNumber++;
        }


    }

    private String[] getNextResult(Iterator<Text> mapperResults) throws IOException {
        if (mapperResults.hasNext()) {
            Text mapperResult = mapperResults.next();
            return KEY_VALUE_DELIMITER.split(mapperResult.toString());
        }
        else {
            return null;
        }
    }

    private int getNumberOfMappers(Iterator<Text> mapperResults) {
        int numberOfMappers = 0;
        while (mapperResults.hasNext()) {
            mapperResults.next();
            numberOfMappers++;
        }
        return numberOfMappers;
    }

   

    public int getNumberOfMappers() {
        return numberOfMappers;
    }


    public double[][] getLambdaGroup() {
        return LambdaGroup;
    }

    public double[][][] getMuGroup() {
        return MuGroup;
    }

    public double[][] getY0kGroup() {
        return Y0kGroup;
    }
    
    public double[][][] getX0kGroup() {
        return X0kGroup;
    }
    
    public String[] getSplitIds() {
        return splitIds;
    }


}
