package com.intentmedia.admm;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.codehaus.jackson.annotate.JsonProperty;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import static com.intentmedia.admm.AdmmIterationHelper.admmReducerContextToJson;
import static com.intentmedia.admm.AdmmIterationHelper.jsonToAdmmReducerContext;

public class AdmmReducerContext implements Writable {

    @JsonProperty("Y0k")
    private double[] Y0k; // Y0k in the paper

    @JsonProperty("X0k")
    private double[][] X0k; // X0k in the paper

    @JsonProperty("Lambda")
    private double[] Lambda; // updated Lambda will be emitted to the reducer

    @JsonProperty("Mu")
    private double[][] Mu; // updated Mu will be emitted to the reducer

    

    public AdmmReducerContext(double[] Lambda, double[][] Mu, double[] Y0k, double[][] X0k) {
        this.Lambda = Lambda;
        this.Mu = Mu;
        this.Y0k = Y0k;
        this.X0k = X0k;
        
        
    }

    public AdmmReducerContext() {
    }

    public void setAdmmReducerContext(AdmmReducerContext context) {
        this.Lambda = context.Lambda;
        this.Mu = context.Mu;
        this.Y0k = context.Y0k;
        this.X0k = context.X0k;
        

        
    }

    @Override
    public void write(DataOutput out) throws IOException {
        Text contextJson = new Text(admmReducerContextToJson(this));
        contextJson.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        Text contextJson = new Text();
        contextJson.readFields(in);
        setAdmmReducerContext(jsonToAdmmReducerContext(contextJson.toString()));
    }

    @JsonProperty("Lambda")
    public double[] getLambda() {
        return Lambda;
    }

    @JsonProperty("Mu")
    public double[][] getMu() {
        return Mu;
    }

    @JsonProperty("X0k")
    public double[][] getX0k() {
        return X0k;
    }

    @JsonProperty("Y0k")
    public double[] getY0k() {
        return Y0k;
    }

   
}
