.. Advanced X-ray Diffraction (aEDXD) User Manual

Welcome to the Amorphous Energy Dispersive X-ray Diffraction (aEDXD) User Manual. This comprehensive guide will lead you through the advanced features of aEDXD, focusing on analyzing complex materials, such as polymers. We will provide you with detailed step-by-step instructions, expert insights, and in-depth information to help you effectively use the software for X-ray diffraction data analysis.

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :numbered:
   
   intro
   basic_data_analysis
   advanced_data_analysis
   access_and_update_aedxd

Introduction to aEDXD
=====================

Advanced X-ray Diffraction (aEDXD) is a powerful tool for gaining deep insights into the atomic and molecular structure of materials, making it invaluable for the analysis of complex materials like polymers. aEDXD helps you uncover detailed information about the arrangement of atoms and molecules, which is crucial for understanding material properties and behavior.

Basic Data Analysis
===================

Data Import
-----------

To start using aEDXD effectively, you must import your X-ray diffraction data. Follow these detailed steps for a smooth data import:

1. Launch aEDXD: Open the aEDXD software on your computer to get started.
2. Import Data: Click on the "File" menu in the aEDXD interface.
3. Choose Data Import Option: From the dropdown menu, choose "Import Data" or a similar option for data import.
4. Select Import Type: In the dialog that appears, you have options to "Add all 2Theta files" or "Add a single 2Theta file."
   - Choose "Add all 2Theta files" if you want to import multiple files collected at different 2Theta angles.
   - Alternatively, select "Add a single 2Theta file" if you want to import a single data file.
5. Locate Data Files: A file explorer window will open. Navigate to the folder containing your data files.
6. Select Files for Import: Select the files you want to import. You can select multiple files if they belong to the same 2Theta angle.
7. Confirm Import: Click "Open" or "Import" to initiate the data import process.
8. Data Organization: The software will automatically organize the files by 2Theta angles. Each set of data files collected at the same 2Theta angle will be grouped together for convenience.

Data Preprocessing
-------------------

Before diving into advanced analyses, it's essential to prepare your data for accuracy. Detailed data preprocessing is essential:

1. Data Quality Check: Carefully inspect the imported data for any irregularities or artifacts within the diffraction pattern. Address any issues before proceeding.
2. Noise Removal: Remove background noise or irrelevant data points that might distort the accuracy of your analysis. Use the software's built-in tools for noise reduction.
3. Correction Algorithms: Apply correction algorithms to account for factors like instrumental broadening or detector sensitivity. This ensures that the data is corrected for any instrument-related artifacts.

Atomic Fraction Calculation
---------------------------

Accurate atomic fraction specification is crucial for materials with known atomic compositions. To calculate atomic fractions for your material, follow these detailed steps:

1. Project Selection: Open the project corresponding to your material within aEDXD, then access the atomic fraction settings.
2. Fraction Specification: Specify the atomic fractions for each element present in your material. Ensure that these fractions sum up to 1, representing the entire material. For instance, SiO2 consists of Si (1/3) and O (2/3).
3. Save Settings: Save the atomic fraction settings, and apply them to your data analysis. This step is fundamental for accurate analysis.

Advanced Data Analysis
=======================

Handling Data Outside the Range
-------------------------------

Complex materials like polymers often exhibit varying molecular structures and diffraction patterns. In-depth knowledge of your data is essential:

1. Reference Research Papers: To understand the unique characteristics of the polymer you're analyzing, consult recent research papers specific to that material. These papers may provide valuable insights into the amorphous diffraction patterns associated with polymers.

Calculating Atomic Fractions for Polymers
-----------------------------------------

Analyzing polymer data involves precise calculation of atomic fractions, considering all elements except hydrogen:

1. Identify Atomic Composition: Determine the atoms present in the polymer's atomic composition. Exclude hydrogen, as it has minimal contribution to diffraction.
2. Fraction Calculation: Calculate the atomic fractions for each element, ensuring they sum to 1. Input these calculated fractions into aEDXD.

Adjusting the High 2-Theta Data
------------------------------

Normalization of data can be affected by the highest 2-Theta data points, particularly in complex materials:

1. Disable High 2-Theta Data: Uncheck the highest 2-Theta data points that don't significantly contribute to your analysis.
2. Reapply Analysis: After removing unnecessary data points, reapply your analysis. Ensure that the resulting data oscillates around a value of 1, which represents the physical behavior of the pair distribution function (PDF) analysis.

Accessing and Updating aEDXD
============================

Downloading aEDXD
-----------------

To get started with aEDXD, follow these steps to download the software:

1. Visit GitHub: Access the aEDXD GitHub repository where you can find the latest version of the software.
2. Download for Windows: If you're using a Windows computer, download the "win64" version. If you're on a Mac, download the "app" version.
3. Access Source Code: If you're interested in exploring the source code and understanding how aEDXD works, it's available in the same repository.

Using aEDXD on Windows and Mac
-------------------------------

Depending on your operating system, choose the appropriate version of aEDXD:

- Windows: Download and install the "win64" version.
- Mac: Download and install the "app" version.

Checking for Updates
-------------------

To stay up-to-date with aEDXD's latest features and bug fixes:

1. Visit the aEDXD GitHub Repository: Periodically check the aEDXD GitHub repository for updates.
2. Download and Install Updates: If updates are available, download and install the latest version to benefit from improved functionality and bug fixes.

By following these steps, you can access and keep aEDXD up-to-date with the latest enhancements and improvements.

This comprehensive guide provides you with detailed instructions, expert insights, and access information to effectively utilize aEDXD for advanced X-ray diffraction data analysis. Keep in mind that precise data analysis and thoughtful handling of your material's atomic composition are essential for accurate results.
