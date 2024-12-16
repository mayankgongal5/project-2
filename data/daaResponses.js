const daaResponses = {
  1: `Theory: Binary Search Using Divide and Conquer

  Binary search is an efficient algorithm for finding an element in a sorted array. It works by dividing the search range into halves repeatedly until the element is found or the range is empty. This is a classic example of the Divide and Conquer approach:
	1.	Divide: Split the array into two halves by calculating the middle index.
	2.	Conquer: Check if the middle element is the target. If not, determine which half to search:
	•	If the target is smaller, search the left half.
	•	If larger, search the right half.
	3.	Combine: The result is directly obtained once the target is found.

  Complexity:
	•	Time Complexity: ￼, as the search range halves with each step.
	•	Space Complexity: ￼ for iterative, ￼ for recursive (due to function call stack).

  Code: Iterative Implementation

  #include <iostream>
  using namespace std;

  // Binary Search Function
  int binarySearch(int arr[], int n, int target) {
      int left = 0, right = n - 1;

      while (left <= right) {
          int mid = left + (right - left) / 2;

          // Check if target is at mid
          if (arr[mid] == target)
              return mid;

          // Decide the side to search
          if (arr[mid] < target)
              left = mid + 1; // Search right half
          else
              right = mid - 1; // Search left half
      }

      return -1; // Target not found
  }

  int main() {
      int arr[] = {2, 4, 6, 8, 10, 12, 14};
      int n = sizeof(arr) / sizeof(arr[0]);
      int target = 8;

      int result = binarySearch(arr, n, target);
      if (result != -1)
          cout << "Element found at index: " << result << endl;
      else
          cout << "Element not found." << endl;

      return 0;
  }

  Explanation of the Code:
	1.	Initialization: Define the search range with left and right.
	2.	Middle Calculation: Compute the midpoint using mid = left + (right - left) / 2 to avoid overflow.
	3.	Condition Check:
	•	If the middle element matches the target, return its index.
	•	Otherwise, adjust left or right to focus on the relevant half.
	4.	Output: Print the index if found or indicate that the element is absent.

  This code is short, simple, and follows the divide-and-conquer approach.`,

  2: `Theory: Merge Sort and Quick Sort

  Merge Sort
	•	Concept: A divide-and-conquer sorting algorithm that splits the array into two halves, sorts each half recursively, and then merges the sorted halves.
	•	Steps:
	1.	Divide the array into two halves.
	2.	Recursively sort each half.
	3.	Merge the sorted halves into a single sorted array.
	•	Complexity:
	•	Time: ￼ (always).
	•	Space: ￼ (due to temporary arrays for merging).

  Quick Sort
	•	Concept: A divide-and-conquer sorting algorithm that picks a pivot, partitions the array into elements smaller and larger than the pivot, and sorts the partitions recursively.
	•	Steps:
	1.	Choose a pivot element.
	2.	Partition the array so that elements less than the pivot are on the left, and greater elements are on the right.
	3.	Recursively sort the left and right partitions.
	•	Complexity:
	•	Time: ￼ (average), ￼ (worst case, if partitioning is unbalanced).
	•	Space: ￼ (recursive stack).

  Code: Merge Sort

  #include <iostream>
  using namespace std;

  // Merge function to combine two sorted halves
  void merge(int arr[], int left, int mid, int right) {
      int n1 = mid - left + 1;
      int n2 = right - mid;

      int L[n1], R[n2];

      // Copy data to temp arrays
      for (int i = 0; i < n1; i++) L[i] = arr[left + i];
      for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

      // Merge the temp arrays
      int i = 0, j = 0, k = left;
      while (i < n1 && j < n2) {
          arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
      }

      // Copy remaining elements
      while (i < n1) arr[k++] = L[i++];
      while (j < n2) arr[k++] = R[j++];
  }

  // Merge Sort function
  void mergeSort(int arr[], int left, int right) {
      if (left < right) {
          int mid = left + (right - left) / 2;

          // Sort each half
          mergeSort(arr, left, mid);
          mergeSort(arr, mid + 1, right);

          // Merge sorted halves
          merge(arr, left, mid, right);
      }
  }

  int main() {
      int arr[] = {12, 11, 13, 5, 6, 7};
      int n = sizeof(arr) / sizeof(arr[0]);

      mergeSort(arr, 0, n - 1);

      cout << "Sorted array: ";
      for (int i = 0; i < n; i++) cout << arr[i] << " ";
      return 0;
  }

  Code: Quick Sort

  #include <iostream>
  using namespace std;

  // Partition function to rearrange elements around pivot
  int partition(int arr[], int low, int high) {
      int pivot = arr[high]; // Choosing the last element as pivot
      int i = low - 1;

      for (int j = low; j < high; j++) {
          if (arr[j] < pivot) {
              i++;
              swap(arr[i], arr[j]);
          }
      }
      swap(arr[i + 1], arr[high]);
      return i + 1;
  }

  // Quick Sort function
  void quickSort(int arr[], int low, int high) {
      if (low < high) {
          int pi = partition(arr, low, high);

          // Recursively sort partitions
          quickSort(arr, low, pi - 1);
          quickSort(arr, pi + 1, high);
      }
  }

  int main() {
      int arr[] = {10, 7, 8, 9, 1, 5};
      int n = sizeof(arr) / sizeof(arr[0]);

      quickSort(arr, 0, n - 1);

      cout << "Sorted array: ";
      for (int i = 0; i < n; i++) cout << arr[i] << " ";
      return 0;
  }

  Explanation of the Code
	1.	Merge Sort:
	•	Recursively splits the array into halves.
	•	Merges sorted halves using the merge() function.
	2.	Quick Sort:
	•	Partitions the array using the partition() function.
	•	Recursively sorts partitions.

  Both algorithms are efficient, but choose based on your use case:
	•	Use Merge Sort for guaranteed ￼ performance and stable sorting.
	•	Use Quick Sort for average ￼ with better in-place sorting. `,

  3: `Theory: Knapsack Problem

  The Knapsack Problem involves selecting items with given weights and values to maximize the total value within a weight limit. There are two types:
	1.	0/1 Knapsack: Each item can be chosen once or not at all.
	2.	Fractional Knapsack: Items can be split (not covered here).

  Approach: Dynamic Programming (DP) for 0/1 Knapsack:
	1.	Create a DP table dp[i][w] where i is the item index and w is the weight limit.
	2.	Recurrence:
	•	Exclude item: dp[i][w] = dp[i-1][w].
	•	Include item: dp[i][w] = value[i] + dp[i-1][w-weight[i]].
	3.	Use maximum of both.

  Code: 0/1 Knapsack

  #include <iostream>
  #include <vector>
  using namespace std;

  int knapsack(int W, vector<int>& wt, vector<int>& val, int n) {
      vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

      for (int i = 1; i <= n; i++) {
          for (int w = 1; w <= W; w++) {
              if (wt[i - 1] <= w)
                  dp[i][w] = max(dp[i - 1][w], val[i - 1] + dp[i - 1][w - wt[i - 1]]);
              else
                  dp[i][w] = dp[i - 1][w];
          }
      }

      return dp[n][W];
  }

  int main() {
      int W = 50;
      vector<int> wt = {10, 20, 30};
      vector<int> val = {60, 100, 120};
      int n = wt.size();

      cout << "Maximum value: " << knapsack(W, wt, val, n) << endl;
      return 0;
  }

  Explanation
	•	Input: W (max weight), wt (weights), val (values).
	•	Output: Maximum achievable value.
	•	Steps:
	1.	Fill the DP table iteratively.
	2.	Use max() to decide between including or excluding an item.`,

  4: `Theory: Kruskal’s Algorithm

  Kruskal’s Algorithm is used to find the Minimum Spanning Tree (MST) of a connected, weighted graph. It follows the greedy approach by selecting the smallest available edge that doesn’t form a cycle.

  Steps:
	1.	Sort all edges by weight in ascending order.
	2.	Use a disjoint-set (Union-Find) to manage connected components.
	3.	For each edge:
	•	Add the edge if it connects two disjoint sets (no cycle).
	4.	Stop when the MST includes ￼ edges (where ￼ is the number of vertices).

  Complexity:
	•	Sorting edges: ￼
	•	Union-Find operations: ￼, where ￼ is the inverse Ackermann function.
	•	Total: ￼

  Code: Kruskal’s Algorithm

  #include <iostream>
  #include <vector>
  #include <algorithm>
  using namespace std;

  // Edge structure
  struct Edge {
      int u, v, weight;
      bool operator<(const Edge& other) const { return weight < other.weight; }
  };

  // Union-Find functions
  int findParent(int node, vector<int>& parent) {
      if (node != parent[node])
          parent[node] = findParent(parent[node], parent); // Path compression
      return parent[node];
  }

  void unionSets(int u, int v, vector<int>& parent, vector<int>& rank) {
      u = findParent(u, parent);
      v = findParent(v, parent);
      if (rank[u] < rank[v])
          parent[u] = v;
      else if (rank[u] > rank[v])
          parent[v] = u;
      else {
          parent[v] = u;
          rank[u]++;
      }
  }

  int kruskal(int n, vector<Edge>& edges) {
      sort(edges.begin(), edges.end()); // Sort edges by weight
      vector<int> parent(n), rank(n, 0);
      for (int i = 0; i < n; i++) parent[i] = i;

      int mstWeight = 0;
      for (const Edge& edge : edges) {
          if (findParent(edge.u, parent) != findParent(edge.v, parent)) {
              mstWeight += edge.weight; // Add edge weight to MST
              unionSets(edge.u, edge.v, parent, rank);
          }
      }
      return mstWeight;
  }

  int main() {
      int n = 4; // Number of vertices
      vector<Edge> edges = {{0, 1, 10}, {0, 2, 6}, {0, 3, 5}, {1, 3, 15}, {2, 3, 4}};

      cout << "Minimum Spanning Tree weight: " << kruskal(n, edges) << endl;
      return 0;
  }

  Explanation
	1.	Edge Structure: Each edge contains two vertices (u and v) and a weight.
	2.	Union-Find:
	•	findParent: Finds the root of a node’s set.
	•	unionSets: Combines two sets using rank to keep the tree flat.
	3.	MST Construction:
	•	Sort edges by weight.
	•	Add edges to the MST if they don’t form a cycle.
	4.	Output: Prints the weight of the MST.

  This code is concise, clear, and efficiently implements Kruskal’s Algorithm.`,

  5: `Theory: Prim’s Algorithm

  Prim’s Algorithm finds the Minimum Spanning Tree (MST) of a connected, weighted graph. It grows the MST one edge at a time by always choosing the smallest edge connecting a vertex in the MST to a vertex outside it.

  Steps:
	1.	Start with any vertex as part of the MST.
	2.	Use a priority queue or a simple array to select the smallest edge connecting the MST to a new vertex.
	3.	Repeat until all vertices are included.

  Complexity:
	•	Using a priority queue: ￼, where ￼ is the number of vertices and ￼ is the number of edges.

  Code: Prim’s Algorithm

  #include <iostream>
  #include <vector>
  #include <queue>
  #include <climits>
  using namespace std;

  int primsAlgorithm(int n, vector<pair<int, int>> adj[]) {
      vector<int> key(n, INT_MAX); // Min edge weight for each vertex
      vector<bool> inMST(n, false); // Track vertices in MST
      priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

      key[0] = 0; // Start from vertex 0
      pq.push({0, 0}); // {weight, vertex}

      int mstWeight = 0;

      while (!pq.empty()) {
          int u = pq.top().second; // Current vertex
          pq.pop();

          if (inMST[u]) continue; // Skip if already in MST
          inMST[u] = true;
          mstWeight += key[u];

          for (auto [v, weight] : adj[u]) {
              if (!inMST[v] && weight < key[v]) {
                  key[v] = weight;
                  pq.push({key[v], v});
              }
          }
      }

      return mstWeight;
  }

  int main() {
      int n = 5; // Number of vertices
      vector<pair<int, int>> adj[5];

      // Adding edges {u, v, weight}
      adj[0].push_back({1, 2});
      adj[0].push_back({3, 6});
      adj[1].push_back({0, 2});
      adj[1].push_back({2, 3});
      adj[1].push_back({3, 8});
      adj[1].push_back({4, 5});
      adj[2].push_back({1, 3});
      adj[2].push_back({4, 7});
      adj[3].push_back({0, 6});
      adj[3].push_back({1, 8});
      adj[4].push_back({1, 5});
      adj[4].push_back({2, 7});

      cout << "Minimum Spanning Tree weight: " << primsAlgorithm(n, adj) << endl;
      return 0;
  }

  Explanation
	1.	Graph Representation: The graph is represented as an adjacency list (adj[]), where each vertex has a list of pairs {neighbor, weight}.
	2.	Algorithm:
	•	Use a priority queue to efficiently find the smallest edge weight.
	•	Track vertices in the MST using inMST[].
	•	Update key[] with the minimum weights for vertices not yet in the MST.
	3.	Output: The total weight of the MST.

  This code is concise and uses a priority queue for efficiency.`,

  6: `Theory: Huffman Coding

  Huffman Coding is a greedy algorithm used for data compression. It assigns variable-length binary codes to input characters, with shorter codes for more frequent characters.

  Steps:
	1.	Create a frequency table for all characters.
	2.	Build a priority queue (min-heap) with nodes representing characters and their frequencies.
	3.	Repeat until the heap has only one node:
	•	Extract the two smallest frequency nodes.
	•	Merge them into a new node with frequency as the sum of the two.
	•	Push the new node back into the heap.
	4.	Generate binary codes by traversing the tree.

  Applications: Text compression (e.g., ZIP files), encoding in communication systems.

  Code: Huffman Coding

  #include <iostream>
  #include <queue>
  #include <unordered_map>
  using namespace std;

  struct Node {
      char ch;
      int freq;
      Node* left;
      Node* right;

      Node(char c, int f) : ch(c), freq(f), left(nullptr), right(nullptr) {}
  };

  // Compare function for priority queue
  struct Compare {
      bool operator()(Node* a, Node* b) {
          return a->freq > b->freq;
      }
  };

  // Generate codes from the Huffman Tree
  void generateCodes(Node* root, string code, unordered_map<char, string>& huffmanCodes) {
      if (!root) return;

      if (!root->left && !root->right) // Leaf node
          huffmanCodes[root->ch] = code;

      generateCodes(root->left, code + "0", huffmanCodes);
      generateCodes(root->right, code + "1", huffmanCodes);
  }

  // Huffman Encoding
  void huffmanEncoding(string text) {
      unordered_map<char, int> freq;
      for (char ch : text) freq[ch]++;

      priority_queue<Node*, vector<Node*>, Compare> pq;
      for (auto [ch, f] : freq) pq.push(new Node(ch, f));

      while (pq.size() > 1) {
          Node* left = pq.top(); pq.pop();
          Node* right = pq.top(); pq.pop();

          Node* merged = new Node('\0', left->freq + right->freq);
          merged->left = left;
          merged->right = right;
          pq.push(merged);
      }

      Node* root = pq.top();
      unordered_map<char, string> huffmanCodes;
      generateCodes(root, "", huffmanCodes);

      cout << "Huffman Codes:\n";
      for (auto [ch, code] : huffmanCodes)
          cout << ch << ": " << code << "\n";
  }

  int main() {
      string text = "huffman coding example";
      huffmanEncoding(text);
      return 0;
  }

  Explanation
	1.	Input: A string of characters.
	2.	Steps:
	•	Compute frequency of each character.
	•	Build a Huffman Tree using a priority queue.
	•	Generate binary codes by traversing the tree.
	3.	Output: Prints the Huffman codes for each character.

  This code is simple and demonstrates Huffman Coding efficiently.`,

  7: `Theory: Dijkstra’s Algorithm

  Dijkstra’s Algorithm finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative weights. It uses a greedy approach and is widely applied in network routing.

  Steps:
	1.	Assign a tentative distance of infinity to all vertices except the source, which gets 0.
	2.	Use a priority queue to always process the vertex with the smallest tentative distance.
	3.	For the current vertex, update distances of its neighbors if a shorter path is found.
	4.	Mark the current vertex as processed and repeat until all vertices are processed.

  Complexity:
	•	Using a priority queue: ￼, where ￼ is the number of vertices and ￼ is the number of edges.

  Code: Dijkstra’s Algorithm

  #include <iostream>
  #include <vector>
  #include <queue>
  #include <climits>
  using namespace std;

  void dijkstra(int n, vector<pair<int, int>> adj[], int src) {
      vector<int> dist(n, INT_MAX);
      dist[src] = 0;

      priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
      pq.push({0, src}); // {distance, vertex}

      while (!pq.empty()) {
          int u = pq.top().second;
          int d = pq.top().first;
          pq.pop();

          if (d > dist[u]) continue; // Skip outdated distance

          for (auto [v, weight] : adj[u]) {
              if (dist[u] + weight < dist[v]) {
                  dist[v] = dist[u] + weight;
                  pq.push({dist[v], v});
              }
          }
      }

      // Output shortest distances
      cout << "Vertex\tDistance from Source\n";
      for (int i = 0; i < n; i++)
          cout << i << "\t" << dist[i] << "\n";
  }

  int main() {
      int n = 5; // Number of vertices
      vector<pair<int, int>> adj[5];

      // Add edges {u, v, weight}
      adj[0].push_back({1, 2});
      adj[0].push_back({3, 6});
      adj[1].push_back({2, 3});
      adj[1].push_back({3, 8});
      adj[1].push_back({4, 5});
      adj[2].push_back({4, 7});
      adj[3].push_back({4, 9});

      int src = 0; // Source vertex
      dijkstra(n, adj, src);
      return 0;
  }

  Explanation
	1.	Graph Representation: The graph is stored as an adjacency list (adj[]), where each vertex has a list of {neighbor, weight} pairs.
	2.	Algorithm:
	•	Use a priority queue to process vertices with the smallest tentative distance.
	•	Relax edges to update distances of neighboring vertices.
	3.	Output: Shortest distance from the source to all vertices.

  This implementation is simple, efficient, and clearly demonstrates Dijkstra’s Algorithm.`,

  8: `Theory: Longest Common Subsequence (LCS)

  The Longest Common Subsequence (LCS) problem finds the length of the longest subsequence common to two strings. A subsequence is a sequence derived by deleting some or no elements without changing the order.

  Approach:
  Use Dynamic Programming with the following steps:
	1.	Create a 2D DP table dp[m+1][n+1] where m and n are the lengths of the two strings.
	2.	Fill the table using this recurrence:
	•	If characters match: dp[i][j] = dp[i-1][j-1] + 1.
	•	Otherwise: dp[i][j] = max(dp[i-1][j], dp[i][j-1]).
	3.	The value dp[m][n] gives the LCS length.

  Time Complexity: ￼, where ￼ and ￼ are the lengths of the two strings.

  Code: LCS using DP

  #include <iostream>
  #include <vector>
  using namespace std;

  int lcs(string s1, string s2) {
      int m = s1.length(), n = s2.length();
      vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

      for (int i = 1; i <= m; i++) {
          for (int j = 1; j <= n; j++) {
              if (s1[i - 1] == s2[j - 1]) // Characters match
                  dp[i][j] = dp[i - 1][j - 1] + 1;
              else
                  dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
          }
      }

      return dp[m][n];
  }

  int main() {
      string s1 = "ABCBDAB", s2 = "BDCAB";
      cout << "Length of LCS: " << lcs(s1, s2) << endl;
      return 0;
  }

  Explanation
	1.	Input: Two strings s1 and s2.
	2.	Output: Length of the longest common subsequence.
	3.	Steps:
	•	Use a DP table to store the length of the LCS for substrings.
	•	Fill the table iteratively based on the recurrence relation.
	4.	Output Example: For s1 = "ABCBDAB" and s2 = "BDCAB", LCS is "BCAB", and the length is 4.

  This code is concise, easy to understand, and demonstrates LCS using DP effectively.`,

  9: `Theory: Bellman-Ford Algorithm

  The Bellman-Ford Algorithm computes the shortest path from a source vertex to all other vertices in a weighted graph, even with negative edge weights. It detects negative weight cycles.

  Steps:
	1.	Initialize the distance of all vertices as infinity (INT_MAX) except the source vertex, which is 0.
	2.	Relax all edges ￼ times (where ￼ is the number of vertices):
	•	Update the distance of a vertex if a shorter path is found through an edge.
	3.	Check for negative weight cycles by iterating through all edges once more. If a shorter path is found, a negative cycle exists.

  Complexity:
	•	Time: ￼, where ￼ is the number of vertices and ￼ is the number of edges.

  Code: Bellman-Ford Algorithm

  #include <iostream>
  #include <vector>
  using namespace std;

  struct Edge {
      int u, v, weight; // Edge from u to v with a given weight
  };

  void bellmanFord(int n, int src, vector<Edge>& edges) {
      vector<int> dist(n, INT_MAX);
      dist[src] = 0;

      // Relax edges (V-1) times
      for (int i = 0; i < n - 1; i++) {
          for (auto edge : edges) {
              if (dist[edge.u] != INT_MAX && dist[edge.u] + edge.weight < dist[edge.v]) {
                  dist[edge.v] = dist[edge.u] + edge.weight;
              }
          }
      }

      // Check for negative weight cycles
      for (auto edge : edges) {
          if (dist[edge.u] != INT_MAX && dist[edge.u] + edge.weight < dist[edge.v]) {
              cout << "Graph contains a negative weight cycle\n";
              return;
          }
      }

      // Print shortest distances
      cout << "Vertex\tDistance from Source\n";
      for (int i = 0; i < n; i++) {
          cout << i << "\t" << dist[i] << "\n";
      }
  }

  int main() {
      int n = 5; // Number of vertices
      vector<Edge> edges = {
          {0, 1, -1}, {0, 2, 4}, {1, 2, 3}, {1, 3, 2}, {1, 4, 2},
          {3, 2, 5}, {3, 1, 1}, {4, 3, -3}
      };

      int src = 0; // Source vertex
      bellmanFord(n, src, edges);
      return 0;
  }

  Explanation
	1.	Input:
	•	n: Number of vertices.
	•	edges: List of edges {u, v, weight}.
	•	src: Source vertex.
	2.	Output:
	•	Shortest distances from src to all vertices.
	•	Detects negative weight cycles.
	3.	Algorithm:
	•	Relax edges ￼ times to compute shortest paths.
	•	Perform one extra iteration to detect negative cycles.

  This code is straightforward, handles negative weights, and efficiently demonstrates the Bellman-Ford algorithm.`,

  10: `Theory: Floyd-Warshall Algorithm

  The Floyd-Warshall Algorithm is a dynamic programming algorithm used to find the shortest paths between all pairs of vertices in a weighted graph. It works for both positive and negative weights (but no negative weight cycles).

  Steps:
	1.	Initialize a distance matrix dist[][], where dist[i][j] is the direct distance between vertex i and vertex j. If no direct edge exists, set it to infinity.
	2.	For each vertex k, update the distance between every pair of vertices i and j by considering if passing through k gives a shorter path:
  dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]).
	3.	Repeat the above process for all vertices as intermediate nodes.

  Complexity:
	•	Time complexity: ￼, where ￼ is the number of vertices.
	•	Space complexity: ￼, for the distance matrix.

  Code: Floyd-Warshall Algorithm

  #include <iostream>
  #include <vector>
  #include <climits>
  using namespace std;

  void floydWarshall(int V, vector<vector<int>>& dist) {
      // Apply Floyd-Warshall Algorithm
      for (int k = 0; k < V; k++) {
          for (int i = 0; i < V; i++) {
              for (int j = 0; j < V; j++) {
                  if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                      dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                  }
              }
          }
      }

      // Print the shortest distance matrix
      cout << "Shortest distances between every pair of vertices:\n";
      for (int i = 0; i < V; i++) {
          for (int j = 0; j < V; j++) {
              if (dist[i][j] == INT_MAX)
                  cout << "INF ";
              else
                  cout << dist[i][j] << " ";
          }
          cout << endl;
      }
  }

  int main() {
      int V = 4; // Number of vertices
      vector<vector<int>> dist = {
          {0, 3, INT_MAX, 7},
          {8, 0, 2, INT_MAX},
          {5, INT_MAX, 0, 1},
          {2, INT_MAX, INT_MAX, 0}
      };

      floydWarshall(V, dist);
      return 0;
  }

  Explanation
	1.	Input:
	•	V: Number of vertices.
	•	dist: 2D array representing the initial distances between vertices. If no edge exists, set it to INT_MAX.
	2.	Output: Shortest distances between every pair of vertices.
	3.	Steps:
	•	Initialize the distance matrix.
	•	Use the three nested loops to update the shortest paths by considering each vertex as an intermediate node.
	•	Print the shortest distance matrix.

  This implementation is simple and effective for computing the shortest paths between all pairs of vertices in a graph using the Floyd-Warshall algorithm.`,
};

module.exports = daaResponses;
