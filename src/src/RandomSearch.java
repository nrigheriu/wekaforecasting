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
 *    RandomSearch.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package src;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;
import java.util.Random;

import weka.filters.supervised.attribute.TSLagMaker;
import weka.classifiers.functions.LinearRegression;
import weka.core.*;

/**
 <!-- globalinfo-start -->
 * SimmulatedAnnealing:<br/>
 * <br/>
 */
public class RandomSearch{
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
    /**
     * Attributes which should always be included in the subset list
     */
    protected ArrayList<Integer> listOfAttributesWhichShouldAlwaysBeThere = new ArrayList<Integer>();

    /** holds the merit of the best subset found */
    protected double m_bestMerit;

    /** holds the maximum size of the lookup cache for evaluated subsets */
    protected int m_cacheSize;

    /**
     * Constructor
     */
    public RandomSearch() {
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
     * Searches the attribute subset space by best first search
     *
     * @param data the training instances.
     * @return an array (not necessarily ordered) of selected attribute indexes
     * @throws Exception if the search can't be completed
     */
    public int[] search(Instances data, TSLagMaker tsLagMaker, List<String> overlayFields) throws Exception {
        m_totalEvals = 0;
        int m_maxEvals = 10;
        PrintWriter errorLog = new PrintWriter(new FileWriter("RA/errorLog.txt", true));
        TSWrapper tsWrapper = new TSWrapper();
        tsWrapper.buildEvaluator(data);
        LinearRegression linearRegression = new LinearRegression();
        linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1 -R 1E-6"));
        tsWrapper.setM_BaseClassifier(linearRegression);
        m_numAttribs = data.numAttributes();
        SubsetHandler subsetHandler = new SubsetHandler();
        subsetHandler.setM_numAttribs(m_numAttribs);
        BitSet best_group;
        best_group = subsetHandler.getStartSet(1);
        double best_merit;
        Hashtable<String, Double> lookForExistingSubsets = new Hashtable<String, Double>();
        // evaluate the initial subset
        subsetHandler.printGroup(best_group);
        best_merit = tsWrapper.evaluateSubset(best_group, tsLagMaker, overlayFields, false);
        m_totalEvals++;
        String subset_string = best_group.toString();
        lookForExistingSubsets.put(subset_string, best_merit);
        System.out.println("Initial group with numAttribs: " + m_numAttribs + "/n");
        System.out.println("Merit: " + best_merit);
        errorLog.println(best_merit);
        while(m_totalEvals < m_maxEvals){
            BitSet s_new = subsetHandler.changeBits((BitSet)best_group.clone(), 1);
            subset_string = s_new.toString();
            if(!lookForExistingSubsets.containsKey(subset_string)){
                double s_new_merit = tsWrapper.evaluateSubset(s_new, tsLagMaker, overlayFields, false);
                m_totalEvals++;
                System.out.println("New merit: " + s_new_merit);
                errorLog.println(s_new_merit);
                lookForExistingSubsets.put(subset_string, s_new_merit);
                if(decisionFunction(best_merit - s_new_merit)){
                    best_group = (BitSet) s_new.clone();
                    best_merit = s_new_merit;
                    System.out.println("New best merit!");
                }
            }
        }
        System.out.println("Best merit:" + best_merit);
        errorLog.println("Best merit:" + best_merit);
        System.out.println(m_totalEvals);
        tsWrapper.evaluateSubset(best_group, tsLagMaker, overlayFields, true);
        errorLog.println(m_totalEvals);
        errorLog.close();
        return attributeList(best_group);
    }
    protected boolean decisionFunction(double difference){
        boolean change = false;
        int i = 0;
        if(difference > 0)
            change = true;
        else
            change = false;
        return change;
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
}
