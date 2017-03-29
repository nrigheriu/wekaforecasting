package src;

import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.BitSet;
import java.util.Hashtable;

/**
 * Created by nrigheriu on 28.03.17.
 */
public class RAThread extends Thread{
    private String threadName, hashedGroup;
    private Instances data;
    private int startAttrib, endAttrib, m_maxEvals;
    private BitSet temp_group;
    private Hashtable<String, Double> lookForExistingSubsets;
    private SubsetHandler subsetHandler;
    private TSWrapper tsWrapper;
    private String subset_string;
    private RAContainer raContainer;
    public RAThread(String threadName, RAContainer raContainer, int m_maxEvals,
                    SubsetHandler subsetHandler, Classifier classifier, Instances data){
        this.threadName = threadName;
        this.raContainer = raContainer;
        this.m_maxEvals = m_maxEvals;
        this.subsetHandler = subsetHandler;
        this.tsWrapper = new TSWrapper();
        this.tsWrapper.setM_BaseClassifier(classifier);
        try {
            this.tsWrapper.buildEvaluator(data);
        }catch (Exception e){
            e.printStackTrace();
        }

    }
    public void run(){
        System.out.println("Running " + threadName);
        try {
            while (raContainer.getM_totalEvals() < m_maxEvals) {
                BitSet s_new = subsetHandler.changeBits((BitSet) raContainer.getBest_group().clone(), 1);
                subset_string = s_new.toString();
                if (!raContainer.getLookForExistingSubsets().containsKey(subset_string)) {
                    double s_new_merit = tsWrapper.evaluateSubset(s_new, raContainer.getTsLagMaker(), raContainer.getOverlayFields(), false);
                    incrementEvals();
                    System.out.println("New merit: " + s_new_merit);
                    raContainer.putInExistingSubsets(subset_string, s_new_merit);
                    if (decisionFunction(raContainer.getBest_merit() - s_new_merit))
                        raContainer.changeBestGroup(s_new, s_new_merit);
                }
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public synchronized void incrementEvals(){
        raContainer.setM_totalEvals(raContainer.getM_totalEvals() + 1);
    }
    protected boolean decisionFunction(double difference){
        boolean change = false;
        if(difference > 0)
            change = true;
        else
            change = false;
        return change;
    }
}
