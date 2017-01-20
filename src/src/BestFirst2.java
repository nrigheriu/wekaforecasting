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
 *    BestFirst2.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package src;

import java.io.Serializable;
import java.util.*;
import java.util.Random;
import weka.attributeSelection.*;
import weka.classifiers.functions.LinearRegression;
import weka.core.*;
import weka.filters.supervised.attribute.TSLagMaker;

/**
 <!-- globalinfo-start --> 
 * BestFirst2:<br/>
 * <br/>
 * Searches the space of attribute subsets by greedy hillclimbing augmented with
 * a backtracking facility. Setting the number of consecutive non-improving
 * nodes allowed controls the level of backtracking done. Best first may start
 * with the empty set of attributes and search forward, or start with the full
 * set of attributes and search backward, or start at any point and search in
 * both directions (by considering all possible single attribute additions and
 * deletions at a given point).<br/>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
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
 * @author Mark Hall (mhall@cs.waikato.ac.nz) Martin Guetlein (cashing merit of
 *         expanded nodes)
 * @version $Revision: 10396 $
 */
public class BestFirst2 implements OptionHandler,
        StartSetHandler {

  /** for serialization */
  static final long serialVersionUID = 7841338689536821867L;

  // Inner classes
  /**
   * Class for a node in a linked list. Used in best first search.
   * 
   * @author Mark Hall (mhall@cs.waikato.ac.nz)
   **/
  public class Link2 implements Serializable, RevisionHandler {

    /** for serialization */
    static final long serialVersionUID = -8236598311516351420L;

    /* BitSet group; */
    Object[] m_data;
    double m_merit;

    /**
     * Constructor
     */
    public Link2(Object[] data, double mer) {
      // group = (BitSet)gr.clone();
      m_data = data;
      m_merit = mer;
    }

    /** Get a group */
    public Object[] getData() {
      return m_data;
    }

    @Override
    public String toString() {
      return ("Node: " + m_data.toString() + "  " + m_merit);
    }

    /**
     * Returns the revision string.
     * 
     * @return the revision
     */
    @Override
    public String getRevision() {
      return RevisionUtils.extract("$Revision: 10396 $");
    }
  }

  /**
   * Class for handling a linked list. Used in best first search. Extends the
   * Vector class.
   * 
   * @author Mark Hall (mhall@cs.waikato.ac.nz)
   **/
  public class LinkedList2 extends ArrayList<Link2> {

    /** for serialization */
    static final long serialVersionUID = 3250538292330398929L;

    /** Max number of elements in the list */
    int m_MaxSize;

    // ================
    // Public methods
    // ================
    public LinkedList2(int sz) {
      super();
      m_MaxSize = sz;
    }

    /**
     * removes an element (Link) at a specific index from the list.
     * 
     * @param index the index of the element to be removed.
     **/
    public void removeLinkAt(int index) throws Exception {

      if ((index >= 0) && (index < size())) {
        remove(index);
      } else {
        throw new Exception("index out of range (removeLinkAt)");
      }
    }

    /**
     * returns the element (Link) at a specific index from the list.
     * 
     * @param index the index of the element to be returned.
     **/
    public Link2 getLinkAt(int index) throws Exception {

      if (size() == 0) {
        throw new Exception("List is empty (getLinkAt)");
      } else {
        if ((index >= 0) && (index < size())) {
          return ((get(index)));
        } else {
          throw new Exception("index out of range (getLinkAt)");
        }
      }
    }

    /**
     * adds an element (Link) to the list.
     * 
     * @param data the attribute set specification
     * @param mer the "merit" of this attribute set
     **/
    public void addToList(Object[] data, double mer) throws Exception {
      Link2 newL = new Link2(data, mer);

      if (size() == 0) {
        add(newL);
      } else {
        if (mer > (get(0)).m_merit) {
          if (size() == m_MaxSize) {
            removeLinkAt(m_MaxSize - 1);
          }

          // ----------
          add(0, newL);
        } else {
          int i = 0;
          int size = size();
          boolean done = false;

          // ------------
          // don't insert if list contains max elements an this
          // is worst than the last
          if ((size == m_MaxSize) && (mer <= get(size() - 1).m_merit)) {

          }
          // ---------------
          else {
            while ((!done) && (i < size)) {
              if (mer > (get(i)).m_merit) {
                if (size == m_MaxSize) {
                  removeLinkAt(m_MaxSize - 1);
                }

                // ---------------------
                add(i, newL);
                done = true;
              } else {
                if (i == size - 1) {
                  add(newL);
                  done = true;
                } else {
                  i++;
                }
              }
            }
          }
        }
      }
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

  // member variables
  /** maximum number of stale nodes before terminating search */
  protected int m_maxStale;

  /** 0 == backward search, 1 == forward search, 2 == bidirectional */
  protected int m_searchDirection;

  /** search direction: backward */
  protected static final int SELECTION_BACKWARD = 0;
  /** search direction: forward */
  protected static final int SELECTION_FORWARD = 1;
  /** search direction: bidirectional */
  protected static final int SELECTION_BIDIRECTIONAL = 2;
  /** search directions */
  public static final Tag[] TAGS_SELECTION = {
    new Tag(SELECTION_BACKWARD, "Backward"),
    new Tag(SELECTION_FORWARD, "Forward"),
    new Tag(SELECTION_BIDIRECTIONAL, "Bi-directional"), };

  /** holds an array of starting attributes */
  protected int[] m_starting;

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
    return "BestFirst2:\n\n"
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
  public BestFirst2() {
    resetOptions();
  }

  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   * 
   **/
  @Override
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
  @Override
  public void setOptions(String[] options) throws Exception {
    String optionString;
    resetOptions();

    optionString = Utils.getOption('P', options);
    if (optionString.length() != 0) {
      setStartSet(optionString);
    }

    optionString = Utils.getOption('D', options);

    if (optionString.length() != 0) {
      setDirection(new SelectedTag(Integer.parseInt(optionString),
        TAGS_SELECTION));
    } else {
      setDirection(new SelectedTag(SELECTION_FORWARD, TAGS_SELECTION));
    }

    optionString = Utils.getOption('N', options);

    if (optionString.length() != 0) {
      setSearchTermination(Integer.parseInt(optionString));
    }

    optionString = Utils.getOption('S', options);
    if (optionString.length() != 0) {
      setLookupCacheSize(Integer.parseInt(optionString));
    }

    m_debug = Utils.getFlag('Z', options);
  }

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
  @Override
  public void setStartSet(String startSet) throws Exception {
    m_startRange.setRanges(startSet);
  }

  /**
   * Returns a list of attributes (and or attribute ranges) as a String
   * 
   * @return a list of attributes (and or attribute ranges)
   */
  @Override
  public String getStartSet() {
    return m_startRange.getRanges();
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String searchTerminationTipText() {
    return "Specify the number of consecutive non-improving nodes to allow "
      + "before terminating the search.";
  }

  /**
   * Set the numnber of non-improving nodes to consider before terminating
   * search.
   * 
   * @param t the number of non-improving nodes
   * @throws Exception if t is less than 1
   */
  public void setSearchTermination(int t) throws Exception {
    if (t < 1) {
      throw new Exception("Value of -N must be > 0.");
    }

    m_maxStale = t;
  }

  /**
   * Get the termination criterion (number of non-improving nodes).
   * 
   * @return the number of non-improving nodes
   */
  public int getSearchTermination() {
    return m_maxStale;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String directionTipText() {
    return "Set the direction of the search.";
  }

  /**
   * Set the search direction
   * 
   * @param d the direction of the search
   */
  public void setDirection(SelectedTag d) {

    if (d.getTags() == TAGS_SELECTION) {
      m_searchDirection = d.getSelectedTag().getID();
    }
  }

  /**
   * Get the search direction
   * 
   * @return the direction of the search
   */
  public SelectedTag getDirection() {

    return new SelectedTag(m_searchDirection, TAGS_SELECTION);
  }

  /**
   * Gets the current settings of BestFirst2.
   * 
   * @return an array of strings suitable for passing to setOptions()
   */
  @Override
  public String[] getOptions() {

    Vector<String> options = new Vector<String>();

    if (!(getStartSet().equals(""))) {
      options.add("-P");
      options.add("" + startSetToString());
    }
    options.add("-D");
    options.add("" + m_searchDirection);
    options.add("-N");
    options.add("" + m_maxStale);

    return options.toArray(new String[0]);
  }

  /**
   * converts the array of starting attributes to a string. This is used by
   * getOptions to return the actual attributes specified as the starting set.
   * This is better than using m_startRanges.getRanges() as the same start set
   * can be specified in different ways from the command line---eg 1,2,3 == 1-3.
   * This is to ensure that stuff that is stored in a database is comparable.
   * 
   * @return a comma seperated list of individual attribute numbers as a String
   */
  private String startSetToString() {
    StringBuffer FString = new StringBuffer();
    boolean didPrint;

    if (m_starting == null) {
      return getStartSet();
    }
    for (int i = 0; i < m_starting.length; i++) {
      didPrint = false;

      if ((m_hasClass == false) || (m_hasClass == true && i != m_classIndex)) {
        FString.append((m_starting[i] + 1));
        didPrint = true;
      }

      if (i == (m_starting.length - 1)) {
        FString.append("");
      } else {
        if (didPrint) {
          FString.append(",");
        }
      }
    }

    return FString.toString();
  }

  /**
   * returns a description of the search as a String
   * 
   * @return a description of the search
   */
  @Override
  public String toString() {
    StringBuffer BfString = new StringBuffer();
    BfString.append("\tBest first.\n\tStart set: ");

    if (m_starting == null) {
      BfString.append("no attributes\n");
    } else {
      BfString.append(startSetToString() + "\n");
    }

    BfString.append("\tSearch direction: ");

    if (m_searchDirection == SELECTION_BACKWARD) {
      BfString.append("backward\n");
    } else {
      if (m_searchDirection == SELECTION_FORWARD) {
        BfString.append("forward\n");
      } else {
        BfString.append("bi-directional\n");
      }
    }

    BfString
      .append("\tStale search after " + m_maxStale + " node expansions\n");
    BfString.append("\tTotal number of subsets evaluated: " + m_totalEvals
      + "\n");
    BfString.append("\tMerit of best subset found: "
      + Utils.doubleToString(Math.abs(m_bestMerit), 8, 3) + "\n");
    return BfString.toString();
  }

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
    TSWrapper tsWrapper = new TSWrapper();
    tsWrapper.buildEvaluator(data);
    LinearRegression linearRegression = new LinearRegression();
    linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1 -R 1E-6"));
    tsWrapper.setM_BaseClassifier(linearRegression);
    m_numAttribs = data.numAttributes();
    BitSet best_group = getStartSet(m_numAttribs, 18), temp_group;
    int temp = 60, initialTemp = temp;
    double best_merit = -Double.MAX_VALUE;
    double merit;
    Hashtable<String, Double> lookup = new Hashtable<String, Double>();
    // evaluate the initial subset
    best_merit = tsWrapper.evaluateSubset(best_group, tsLagMaker, overlayFields);
    String subset_string = best_group.toString();
    lookup.put(subset_string, best_merit);
    System.out.println("Initial group with numAttribs: " + m_numAttribs + "/n");
    printGroup(best_group, m_numAttribs);
    System.out.println("Merit: " + best_merit);
    while(temp > 0){
      BitSet s_new = changeBits(m_numAttribs, best_group);
      subset_string = s_new.toString();
      if(!lookup.containsKey(subset_string)){
          double s_new_merit = tsWrapper.evaluateSubset(s_new, tsLagMaker, overlayFields);
          System.out.println("Changed group: ");
          printGroup(s_new, m_numAttribs);
          System.out.println("New merit: " + s_new_merit);
          lookup.put(subset_string, s_new_merit);
          if(decisionFunction(best_merit - s_new_merit, temp, best_merit, initialTemp)){
            best_group = (BitSet) s_new.clone();
            best_merit = s_new_merit;
          }
      }
      temp --;
    }

    return attributeList(best_group);
  }
  protected boolean decisionFunction(double difference, int temp, double bestMerit, int initialTemp){
      boolean change = false;
      double randomNr = Math.random();
      System.out.println("Difference : " + difference + " Temp: " + temp + " Randomnr: " + randomNr);
      if(difference > 0)
          change = true;
      else{
          double tempPercentage = ((double) temp/initialTemp)*100;
          double errorPercentage = (difference*100)/bestMerit;
          double expFunction = Math.exp(errorPercentage/tempPercentage);
          System.out.println("Expfunction: " + expFunction + "Temp %: " + tempPercentage + "Error % : " + errorPercentage);
          if(expFunction >= randomNr){
              change = true;
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
    boolean includesMoreThan25Percent = false;
    bitSet.set(0); bitSet.set(1);                         //we always need the time stamp feed and the field to be forecasted; the time stamp field may be substituted later by time_remapped by the forecaster anyway
    bitSet.set(numAttribs-1); bitSet.set(numAttribs-2);    //always setting local time remapped powers (for now at least)
      while(!includesMoreThan25Percent) {
      for (int i = 2; i < numAttribs-2; i++) {
        int chance = r.nextInt(100);
        if (chance <= setPercentage) {
          bitSet.set(i);
        }
      }
      if(includesMoreThan25PercentOfFeatures(bitSet, numAttribs))
        includesMoreThan25Percent = true;
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
      boolean includesMoreThan25Percent = false;
      while(!includesMoreThan25Percent) {
        for (int i = 2; i < numAttribs-2; i++) {              //starting from 2 because we need the time stamp and active_power attributes and not changing local time remapped products
          int  chance = r.nextInt(100);
          if (chance <= 9) {
            if (bitSet.get(i))
              bitSet.set(i, false);
            else
              bitSet.set(i, true);
          }
            if(includesMoreThan25PercentOfFeatures(bitSet, numAttribs))         //making sure we dont drop below 25% of Features included after the mutation
              includesMoreThan25Percent = true;
        }
      }
    return bitSet;
  }
  protected boolean includesMoreThan25PercentOfFeatures(BitSet bitSet, int numAttribs){
      int trueBits = 0;
      for (int i = 2; i < numAttribs-2; i++) {              //We count as features just lags and overlay fields, the others are always there
          if(bitSet.get(i))
              trueBits++;
      }
      if((float)trueBits/numAttribs > 0.25)
          return true;
      return  false;
  }

  /**
   * Reset options to default values
   */
  protected void resetOptions() {
    m_maxStale = 5;
    m_searchDirection = SELECTION_FORWARD;
    m_starting = null;
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
