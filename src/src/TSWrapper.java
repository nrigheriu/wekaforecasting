package src;

import com.sun.corba.se.impl.io.TypeMismatchException;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.attribute.TSLagMaker;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Created by cycle on 16.01.2017.
 */
public class TSWrapper {
    private Instances m_data;
    private int m_numAttribs;
    private Classifier m_BaseClassifier;
    private int m_seed;
    private double m_threshold;
    private String m_EvaluationMeasure;
    private int m_classIndex;

    public String getM_EvaluationMeasure() {
        return m_EvaluationMeasure;
    }

    public void setM_EvaluationMeasure(String m_EvaluationMeasure) throws Exception{
        System.out.println("evalmeasure: " + m_EvaluationMeasure);
        if(m_EvaluationMeasure == "RMSE" || m_EvaluationMeasure == "MAPE")
            this.m_EvaluationMeasure = m_EvaluationMeasure;
        else
            throw new TypeMismatchException();
    }

    public TSWrapper(){
        resetOptions();
    }
    @Override
    public String toString(){
        StringBuffer text = new StringBuffer();
        if(m_data == null)
            text.append("\tWrapper subset evaluator has not been built yet\n");
        else
            text.append("\tWrapper Subset Evaluator\n");
        return text.toString();
    }
    public double evaluateSubset(BitSet subset, TSLagMaker tsLagMaker, List<String> overlayFields, boolean testBestModel) throws Exception{
        double returnValue = 0;     //this is error for wrapper and best model mode and bias for bias mode
        int numAttributes = 0;
        int i, j;
        boolean addedClassIndex = false;
        Instances trainCopy;
        Remove delTransform = new Remove();
        List<String> remainingAttributes = new ArrayList<String>();
        String remainingLags = "";
        List<String> newOverlayFields = new ArrayList<String>();
        delTransform.setInvertSelection(true);
        // count attributes set in the BitSet
        for (i = 0; i < m_numAttribs; i++)
            if (subset.get(i))
                numAttributes++;
        // set up an array of attribute indexes for the filter (+1 for the class)
        int[] featArray = new int[numAttributes];
        for (i = 0, j = 0; i < m_numAttribs; i++)
            if (subset.get(i))
                featArray[j++] = i;
        delTransform.setAttributeIndicesArray(featArray);
        delTransform.setInputFormat(m_data);
        trainCopy = Filter.useFilter(m_data, delTransform);

        for (int k = 0; k < trainCopy.numAttributes(); k++) {
            String attrName = trainCopy.attribute(k).name();
            remainingAttributes.add(k, attrName);
            if(attrName.contains("Lag_" + tsLagMaker.getFieldsToLagAsString())){                      //it means we have a lag attribute in the subset and we should update the lagmaker with it
                remainingLags += attrName.substring(attrName.length() - 4, attrName.length()).replaceAll("[^0-9]", "");
                remainingLags += ", ";
            }
        }
        if(!remainingLags.isEmpty())
            remainingLags = remainingLags.substring(0, remainingLags.length()-2);
        else if(remainingLags.isEmpty())
            remainingLags += "11";                                                                //we need at least one lag or else the classifier doesn't have what to train on, this is chosen to be the "worst" so it doesnt influence rest of evalution
        System.out.println("Remaining lags: " + remainingLags);
        tsLagMaker.setLagRange(remainingLags);
        i = 0;
        for (int k = 0; k < overlayFields.size(); k++)                       //updating the tsLagmaker with the still available overlay Fields
            if(remainingAttributes.contains(overlayFields.get(k)))
                newOverlayFields.add(i++, overlayFields.get(k));
        tsLagMaker.setOverlayFields(newOverlayFields);
        TSCV tscv = new TSCV();
        tscv.crossValidateTS(trainCopy, m_BaseClassifier, tsLagMaker, testBestModel);
        returnValue = tscv.calculateErrors(testBestModel,  m_EvaluationMeasure);
        return returnValue;
    }
    public void buildEvaluator(Instances data) throws Exception{
        m_data = data;
        m_classIndex = m_data.classIndex();
        m_numAttribs = m_data.numAttributes();
    }

    public Instances getM_data() {
        return m_data;
    }

    public void setM_data(Instances m_data) {
        this.m_data = m_data;
    }

    public int getM_numAttribs() {
        return m_numAttribs;
    }


    public Classifier getM_BaseClassifier() {
        return this.m_BaseClassifier;
    }

    public void setM_BaseClassifier(Classifier m_BaseClassifier) {
        this.m_BaseClassifier = m_BaseClassifier;
    }

    public int getM_seed() {
        return m_seed;
    }

    public void setM_seed(int m_seed) {
        this.m_seed = m_seed;
    }

    public double getM_threshold() {
        return m_threshold;
    }

    public void setM_threshold(double m_threshold) {
        this.m_threshold = m_threshold;
    }

    protected void resetOptions(){
        m_data = null;
        m_BaseClassifier = new LinearRegression();
        m_seed = 1;
        m_threshold = 0.01;
    }

}
