package src;

import weka.core.Instances;
import weka.filters.supervised.attribute.TSLagMaker;

import java.util.BitSet;
import java.util.Hashtable;
import java.util.List;

/**
 * Created by nrigheriu on 28.03.17.
 */
public class RAContainer {

    private int m_totalEvals;
    private Instances data;
    private int startAttrib, endAttrib, size;
    private BitSet s_new, best_group;
    private Hashtable<String, Double> lookForExistingSubsets;
    private SubsetHandler subsetHandler;
    double best_merit;
    TSLagMaker tsLagMaker;
    List<String> overlayFields;
    public RAContainer(int m_totalEvals, BitSet best_group, Hashtable<String, Double> lookForExistingSubsets,
                       SubsetHandler subsetHandler,  double best_merit, TSLagMaker tsLagMaker, List<String> overlayFields){
        this.m_totalEvals = m_totalEvals;
        this.best_group = best_group;
        this.lookForExistingSubsets = lookForExistingSubsets;
        this.subsetHandler = subsetHandler;
        this.best_merit = best_merit;
        this.tsLagMaker = tsLagMaker;
        this.overlayFields = overlayFields;
    }


    public int getM_totalEvals() {
        return m_totalEvals;
    }

    public void setM_totalEvals(int m_totalEvals) {
        this.m_totalEvals = m_totalEvals;
    }
    public synchronized void putInExistingSubsets(String subset_string, double s_new_merit){
        this.lookForExistingSubsets.put(subset_string, s_new_merit);
    }
    public synchronized void changeBestGroup(BitSet s_new, double s_new_merit){
        this.best_group = (BitSet) s_new.clone();
        this.best_merit = s_new_merit;
        System.out.println("New best merit!");
    }

    public int getStartAttrib() {
        return startAttrib;
    }

    public void setStartAttrib(int startAttrib) {
        this.startAttrib = startAttrib;
    }

    public int getEndAttrib() {
        return endAttrib;
    }

    public void setEndAttrib(int endAttrib) {
        this.endAttrib = endAttrib;
    }

    public int getSize() {
        return size;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public BitSet getS_new() {
        return s_new;
    }

    public void setS_new(BitSet s_new) {
        this.s_new = s_new;
    }

    public BitSet getBest_group() {
        return best_group;
    }

    public void setBest_group(BitSet best_group) {
        this.best_group = best_group;
    }

    public synchronized Hashtable<String, Double> getLookForExistingSubsets() {
        return lookForExistingSubsets;
    }

    public void setLookForExistingSubsets(Hashtable<String, Double> lookForExistingSubsets) {
        this.lookForExistingSubsets = lookForExistingSubsets;
    }

    public SubsetHandler getSubsetHandler() {
        return subsetHandler;
    }

    public void setSubsetHandler(SubsetHandler subsetHandler) {
        this.subsetHandler = subsetHandler;
    }


    public double getBest_merit() {
        return best_merit;
    }

    public void setBest_merit(double best_merit) {
        this.best_merit = best_merit;
    }

    public TSLagMaker getTsLagMaker() {
        return tsLagMaker;
    }

    public void setTsLagMaker(TSLagMaker tsLagMaker) {
        this.tsLagMaker = tsLagMaker;
    }

    public List<String> getOverlayFields() {
        return overlayFields;
    }

    public void setOverlayFields(List<String> overlayFields) {
        this.overlayFields = overlayFields;
    }

}
