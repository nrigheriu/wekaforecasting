/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    SimmulatedAnnealing.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package src;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.*;
import java.util.Random;
import weka.attributeSelection.*;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.supervised.attribute.TSLagMaker;
import weka.filters.unsupervised.attribute.Normalize;
/**
 <!-- globalinfo-start --> 
 * SimmulatedAnnealing:<br/>
 * <br/>
 */
public class SimmulatedAnnealing{

  public class TheVeryBest{
    public BitSet subset = null;
    public Double merit = null;
    public TheVeryBest(BitSet subset, Double merit){
      this.subset = subset;
      this.merit = merit;
    }
    public BitSet getSubset(){
        return this.subset;
    }
    public Double getMerit(){
        return this.merit;
    }
    public void setNewSet(BitSet subset, Double merit){
        this.merit = merit;
        this.subset = subset;
    }
  }

  /** holds the start set for the search as a Range */
  protected Range m_startRange;

  /** does the data have a class */
  protected boolean m_hasClass;

  /** holds the class index */
  protected int m_classIndex;

  /** number of attributes in the data */
  protected int m_numAttribs;

  /** total number of subsets evaluated during a search */
  protected int m_totalEvals;

  /** for debugging */
  protected boolean m_debug;

  /** holds the merit of the best subset found */
  protected double m_bestMerit;

  /** holds the maximum size of the lookup cache for evaluated subsets */
  protected int m_cacheSize;

  /**
   * Returns a string describing this search method
   * 
   * @return a description of the search method suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "SimmulatedAnnealing:\n\n"
      + "Searches the space of attribute subsets by greedy hillclimbing "
      + "augmented with a backtracking facility. Setting the number of "
      + "consecutive non-improving nodes allowed controls the level of "
      + "backtracking done. Best first may start with the empty set of "
      + "attributes and search forward, or start with the full set of "
      + "attributes and search backward, or start at any point and search "
      + "in both directions (by considering all possible single attribute "
      + "additions and deletions at a given point).\n";
  }

  /**
   * Constructor
   */
  public SimmulatedAnnealing() {
    resetOptions();
  }

  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   * 
   **/
  public Enumeration<Option> listOptions() {
    Vector<Option> newVector = new Vector<Option>(4);

    newVector.addElement(new Option("\tSpecify a starting set of attributes."
      + "\n\tEg. 1,3,5-7.", "P", 1, "-P <start set>"));
    newVector.addElement(new Option("\tDirection of search. (default = 1).",
      "D", 1, "-D <0 = backward | 1 = forward " + "| 2 = bi-directional>"));
    newVector.addElement(new Option("\tNumber of non-improving nodes to"
      + "\n\tconsider before terminating search.", "N", 1, "-N <num>"));
    newVector.addElement(new Option(
      "\tSize of lookup cache for evaluated subsets."
        + "\n\tExpressed as a multiple of the number of"
        + "\n\tattributes in the data set. (default = 1)", "S", 1, "-S <num>"));

    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   * <p/>
   * 
   <!-- options-start --> 
   * Valid options are:
   * <p/>
   * 
   * <pre>
   * -P &lt;start set&gt;
   *  Specify a starting set of attributes.
   *  Eg. 1,3,5-7.
   * </pre>
   * 
   * <pre>
   * -D &lt;0 = backward | 1 = forward | 2 = bi-directional&gt;
   *  Direction of search. (default = 1).
   * </pre>
   * 
   * <pre>
   * -N &lt;num&gt;
   *  Number of non-improving nodes to
   *  consider before terminating search.
   * </pre>
   * 
   * <pre>
   * -S &lt;num&gt;
   *  Size of lookup cache for evaluated subsets.
   *  Expressed as a multiple of the number of
   *  attributes in the data set. (default = 1)
   * </pre>
   * 
   <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   * 
   **/

  /**
   * Set the maximum size of the evaluated subset cache (hashtable). This is
   * expressed as a multiplier for the number of attributes in the data set.
   * (default = 1).
   * 
   * @param size the maximum size of the hashtable
   */
  public void setLookupCacheSize(int size) {
    if (size >= 0) {
      m_cacheSize = size;
    }
  }

  /**
   * Return the maximum size of the evaluated subset cache (expressed as a
   * multiplier for the number of attributes in a data set.
   * 
   * @return the maximum size of the hashtable.
   */
  public int getLookupCacheSize() {
    return m_cacheSize;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String lookupCacheSizeTipText() {
    return "Set the maximum size of the lookup cache of evaluated subsets. This is "
      + "expressed as a multiplier of the number of attributes in the data set. "
      + "(default = 1).";
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String startSetTipText() {
    return "Set the start point for the search. This is specified as a comma "
      + "seperated list off attribute indexes starting at 1. It can include "
      + "ranges. Eg. 1,2,5-9,17.";
  }

  /**
   * Sets a starting set of attributes for the search. It is the search method's
   * responsibility to report this start set (if any) in its toString() method.
   * 
   * @param startSet a string containing a list of attributes (and or ranges),
   *          eg. 1,2,6,10-15.
   * @throws Exception if start set can't be set.
   */
  public void setStartSet(String startSet) throws Exception {
    m_startRange.setRanges(startSet);
  }

  /**
   * Returns a list of attributes (and or attribute ranges) as a String
   * 
   * @return a list of attributes (and or attribute ranges)
   */
  protected void printGroup(BitSet tt, int numAttribs) {
    int i;
    for (i = 0; i < numAttribs; i++) {
      if (tt.get(i) == true) {
        System.out.print((i + 1) + " ");
      }
    }
    System.out.println();
  }

  /**
   * Searches the attribute subset space by best first search
   *
   * @param data the training instances.
   * @return an array (not necessarily ordered) of selected attribute indexes
   * @throws Exception if the search can't be completed
   */
  public int[] search(Instances data, TSLagMaker tsLagMaker, List<String> overlayFields) throws Exception {
    m_totalEvals = 0;
    int m_totalEvals = 0;
    PrintWriter errorLog = new PrintWriter(new FileWriter("/home/cycle/workspace/wekaforecasting-new-features/errorLog.txt", true));
    TSWrapper tsWrapper = new TSWrapper();
    tsWrapper.buildEvaluator(data);
    LinearRegression linearRegression = new LinearRegression();
    linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1 -R 1E-6"));
    FilteredClassifier fc = new FilteredClassifier();
    MultiFilter mf = new MultiFilter();
    Normalize normalize = new Normalize();
    NominalToBinary nbt = new NominalToBinary();
    mf.setFilters(new Filter[]{normalize, nbt});
    mf.setInputFormat(data);
    fc.setFilter(mf);
    fc.setClassifier(linearRegression);
    tsWrapper.setM_BaseClassifier(fc);
    //tsWrapper.setM_BaseClassifier(linearRegression);
    m_numAttribs = data.numAttributes();
    BitSet best_group = new BitSet(m_numAttribs), temp_group;
    best_group = getStartSet(m_numAttribs, 30);
    float temp = 7, initialTemp = temp;
    double best_merit = -Double.MAX_VALUE;
    double merit; int i = 0, counter = 0;
    Hashtable<String, Double> lookup = new Hashtable<String, Double>();
    // evaluate the initial subset
    printGroup(best_group, m_numAttribs);
    best_merit = tsWrapper.evaluateSubset(best_group, tsLagMaker, overlayFields);
    m_totalEvals++;
    String subset_string = best_group.toString();
    lookup.put(subset_string, best_merit);
    System.out.println("Initial group with numAttribs: " + m_numAttribs + " temp: " + temp + "/n");
    System.out.println("Merit: " + best_merit);
    errorLog.println(best_merit);
    errorLog.println(temp);
    TheVeryBest theVeryBest = new TheVeryBest((BitSet)best_group.clone(), best_merit);
    boolean[] changedAlthoughWorse = new boolean[9];
    for (int j = 0; j < changedAlthoughWorse.length; j++) {
      changedAlthoughWorse[j] = true;
    }
    while(temp > 0.0001){
      counter = 0;
      BitSet s_new = changeBits(m_numAttribs, (BitSet)best_group.clone());
      subset_string = s_new.toString();
      if(!lookup.containsKey(subset_string)){
          double s_new_merit = tsWrapper.evaluateSubset(s_new, tsLagMaker, overlayFields);
          m_totalEvals++;
          System.out.println("New merit: " + s_new_merit);
          lookup.put(subset_string, s_new_merit);
          if(decisionFunction(best_merit - s_new_merit, temp, best_merit, initialTemp, errorLog)){
            if(best_merit - s_new_merit > 0)
              changedAlthoughWorse[i++] = true;
            best_group = (BitSet) s_new.clone();
            best_merit = s_new_merit;
            errorLog.println(s_new_merit);
            errorLog.println(temp);
          }else
            changedAlthoughWorse[i++] = false;
          if(best_merit < theVeryBest.getMerit()){
              theVeryBest.setNewSet((BitSet) best_group.clone(), best_merit);
          }
          if(temp > 5)
            temp = temp/(float)(1+0.0006*(m_totalEvals-1));
            //temp *= 0.997;
          else
            temp = temp/(float)(1+0.0006*(m_totalEvals-1));

        // temp *= 0.97;
        for (int j = 0; j < changedAlthoughWorse.length; j++)
          if(changedAlthoughWorse[j])
            counter++;
        System.out.println("Percentage of worse sets accepted: " + (float)counter*100/changedAlthoughWorse.length);
        i = i % changedAlthoughWorse.length;
      }
    }
      System.out.println("Best merit: " + theVeryBest.getMerit());
    System.out.println(m_totalEvals);
    printGroup(theVeryBest.getSubset(), m_numAttribs);
    includesMoreThanXPercentOfFeatures(theVeryBest.getSubset(), m_numAttribs, true);
    errorLog.close();
    return attributeList(theVeryBest.getSubset());
  }
  protected float howMuchPercentOfBitsAreDifferent(BitSet bitSet1, BitSet bitSet2, int m_numAttribs){
    float percent = 0;
    int differencesCounter = 0;
    for (int i = 0; i < m_numAttribs; i++) {
      if((bitSet1.get(i) & !bitSet2.get(i)) || (!bitSet1.get(i)) && bitSet2.get(i))
        differencesCounter++;
    }
    return (float) differencesCounter*100/(m_numAttribs-4);
  }
  protected boolean decisionFunction(double difference, float temp, double bestMerit, float initialTemp, PrintWriter errorLog){
      boolean change = false;
      double randomNr = Math.random();
      int i = 0;
      System.out.println("Difference : " + difference + " Temp: " + temp + " Randomnr: " + randomNr);
      if(difference > 0)
          change = true;
      else{
          double tempPercentage = ((double) temp/initialTemp)*100;
          double errorPercentage = (difference*100)/bestMerit;
          double expFunction = Math.exp(errorPercentage/temp);
          System.out.println("Expfunction: " + expFunction + " Temp %: " + temp + " Error% : " + errorPercentage);
          if(expFunction >= randomNr){
              change = true;
              //errorLog.println("Changed!");
              System.out.println("Decided to change to a worse subset!");
          }
      }
      return change;
  }
  /**
   *
   * @param numAttribs size of the BitSet
   *@param setPercentage Percentage of bits randomly being set to 1 for a start
   * @return BitSet with setPercentage of Bits set to 1
   */
  protected BitSet getStartSet (int numAttribs, int setPercentage){
    BitSet bitSet = new BitSet(numAttribs);
    Random r = new Random();
    boolean includesMoreThanXPercent = false;
    bitSet.set(0); bitSet.set(1);                         //we always need the time stamp feed and the field to be forecasted; the time stamp field may be substituted later by time_remapped by the forecaster anyway
    bitSet.set(numAttribs-1); bitSet.set(numAttribs-2);    //always setting local time remapped powers (for now at least)
      while(!includesMoreThanXPercent) {
      for (int i = 2; i < numAttribs-2; i++) {
        int chance = r.nextInt(100);
        if (chance <= setPercentage) {
          bitSet.set(i);
        }
      }
      if(includesMoreThanXPercentOfFeatures(bitSet, numAttribs, false))
        includesMoreThanXPercent = true;
    }
    return bitSet;
  }

    /**
     * changes 2 percent of the bits in the bitset
     * @param numAttribs number of attributes (bits) to consider
     * @param bitSet the bitset to change
     * @return the slightly mutated bitset
     */
  protected BitSet changeBits(int numAttribs, BitSet bitSet){
      Random r = new Random();
      boolean includesMoreThanXPercent = false;
      while(!includesMoreThanXPercent) {
        for (int i = 2; i < numAttribs-2; i++) {              //starting from 2 because we need the time stamp and active_power attributes and not changing local time remapped products
          int  chance = r.nextInt(100);
          if (chance < 2) {
            if (bitSet.get(i))
              bitSet.set(i, false);
            else
              bitSet.set(i, true);
          }
            if(includesMoreThanXPercentOfFeatures(bitSet, numAttribs, false))         //making sure we dont drop below 25% of Features included after the mutation
              includesMoreThanXPercent = true;
            else
                System.out.println("Repeating loop because it doesnt include X% features!");

        }
      }
    return bitSet;
  }
  protected boolean includesMoreThanXPercentOfFeatures(BitSet bitSet, int numAttribs, boolean print){
      int trueBits = 0;
      for (int i = 2; i < numAttribs-2; i++) {              //We count as features just lags and overlay fields, the others are always there
          if(bitSet.get(i))
              trueBits++;
      }
      if (print)
        System.out.println("Including:" + (float)trueBits/numAttribs + "% of features.");
    if((float)trueBits/numAttribs > 0.12)
          return true;
      return  false;
  }

  /**
   * Reset options to default values
   */
  protected void resetOptions() {
    m_startRange = new Range();
    m_classIndex = -1;
    m_totalEvals = 0;
    m_cacheSize = 1;
    m_debug = false;
  }

  /**
   * converts a BitSet into a list of attribute indexes
   * 
   * @param group the BitSet to convert
   * @return an array of attribute indexes
   **/
  protected int[] attributeList(BitSet group) {
    int count = 0;

    // count how many were selected
    for (int i = 0; i < m_numAttribs; i++) {
      if (group.get(i)) {
        count++;
      }
    }

    int[] list = new int[count];
    count = 0;

    for (int i = 0; i < m_numAttribs; i++) {
      if (group.get(i)) {
        list[count++] = i;
      }
    }

    return list;
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 10396 $");
  }
}
