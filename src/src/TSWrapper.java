package src;

import com.sun.corba.se.impl.io.TypeMismatchException;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.BitSet;

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
    public static final int EVAL_RMSE = 1;
    public static final int EVAL_MAPE = 2;
    private BestFirst2 search = new BestFirst2();

    public String getM_EvaluationMeasure() {
        return m_EvaluationMeasure;
    }

    public void setM_EvaluationMeasure(String m_EvaluationMeasure) throws Exception{
        if(m_EvaluationMeasure !="RMSE" || m_EvaluationMeasure !="MAPE"){
            throw new TypeMismatchException();
        }else{
            this.m_EvaluationMeasure = m_EvaluationMeasure;
        }
    }

    public TSWrapper(){
        resetOptions();
    }
    @Override
    public String toString(){
        StringBuffer text = new StringBuffer();
        if(m_data == null){
            text.append("\tWrapper subset evaluator has not been built yet\n");
        }else{
            text.append("\tWrapper Subset Evaluator\n");
        }
        return text.toString();
    }
    public double evaluateSubset(BitSet subset)throws Exception{
        double error = 0;
        int numAttributes = 0;
        int i, j;
        boolean addedClassIndex = false;
        Instances trainCopy = new Instances(m_data);
        Remove delTransform = new Remove();
        delTransform.setInvertSelection(true);
        // count attributes set in the BitSet
        for (i = 0; i < m_numAttribs; i++) {
            if (subset.get(i)) {
                numAttributes++;
            }
        }
        // set up an array of attribute indexes for the filter (+1 for the class)
        int[] featArray = new int[numAttributes + 1];
        System.out.println("Printing subset: ");
        for (int k = 0; k < m_numAttribs; k++) {
            System.out.println("K: " + k + "subset: " + subset.get(k));
        }
        for (i = 0, j = 0; i < m_numAttribs; i++) {
            if (subset.get(i)) {
                if(i == m_classIndex)
                    addedClassIndex = true;
                featArray[j++] = i;
            }
        }
        if(!addedClassIndex)
            featArray[j] = m_classIndex;
        else{                                                           //deleting the last place in the array
            int[] tempArray = new int[featArray.length-1];
            for (int k = 0; k < tempArray.length; k++) {
                tempArray[0] = featArray[k];
                featArray = new int[tempArray.length];
                featArray = tempArray;
            }
        }
        //featArray[j] = m_classIndex;
        /*System.out.println("M class: " + m_classIndex);
        for (int k = 0; k< featArray.length ; k++) {
            System.out.println("K: " + k + " featArray: " + featArray[k]);
        }
        delTransform.setAttributeIndicesArray(featArray);
        delTransform.setInputFormat(trainCopy);
        trainCopy = Filter.useFilter(trainCopy, delTransform);
        System.out.println("traincop size: " + trainCopy.;*/
        for (int k = 0; k < trainCopy.numAttributes(); k++) {
            System.out.println("Before deleteion: " + trainCopy.attribute(k).name());
        }
        trainCopy.deleteAttributeAt(0);
        trainCopy.deleteAttributeAt(1);
        trainCopy.deleteAttributeAt(2);

        for (int k = 0; k < trainCopy.numAttributes(); k++) {
            System.out.println(trainCopy.attribute(k).name());
        }
        TSCV tscv = new TSCV();
        tscv.crossValidateTS(trainCopy, m_BaseClassifier);
        error = tscv.calculateErrors(true);
        return error;
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
        return m_BaseClassifier;
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
