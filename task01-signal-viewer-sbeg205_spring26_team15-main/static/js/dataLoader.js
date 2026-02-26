/**
 * dataLoader.js — Parse CSV/TSV files
 */
const DataLoader = {
  parseCSV(file) {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => resolve(results.data),
        error: (err) => reject(err),
      });
    });
  },

  getBacteriaColumns(data) {
    const metaCols = ['SampleID', 'PatientID', 'Age', 'Sex', 'BMI', 'BodySite', 'Diagnosis'];
    return Object.keys(data[0]).filter((k) => !metaCols.includes(k));
  },
};