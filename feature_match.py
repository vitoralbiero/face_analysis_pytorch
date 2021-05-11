import argparse
from datetime import datetime
from os import makedirs, path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Matcher:
    def __init__(self, probe_path, gallery_path, dataset_name):
        # lenght of ids to get from feature files
        self.id_length = -1
        self.dataset_name = dataset_name

        # load features, subject ids, feature labels from probe file
        probe_file = np.sort(np.loadtxt(probe_path, dtype=np.str))
        self.probe, self.probe_ids, self.probe_labels = self.get_features(probe_file)

        if gallery_path is not None:
            print(f"Matching {probe_path} to {gallery_path}")
            gallery_file = np.sort(np.loadtxt(args.gallery, dtype=np.str))
            # if matching different files, load gallery features, ids and labels
            self.probe_equal_gallery = False
            self.gallery, self.gallery_ids, self.gallery_labels = self.get_features(
                gallery_file
            )
        else:
            print(f"Matching {probe_path} to {probe_path}")
            # if matching to the same file, just create a simbolic link to save memory
            self.probe_equal_gallery = True
            self.gallery = self.probe
            self.gallery_ids = self.probe_ids
            self.gallery_labels = self.probe_labels

        # initiate a matrix NxM with zeros representing impostor matches
        self.authentic_impostor = np.zeros(shape=(len(self.probe), len(self.gallery)))
        for i in range(len(self.probe)):
            # convert authentic matches to 1
            self.authentic_impostor[i, self.probe_ids[i] == self.gallery_ids] = 1

            # remove same feature files
            self.authentic_impostor[i, self.probe_labels[i] == self.gallery_labels] = -1

            if gallery_path is None:
                # remove duplicate matches if matching probe to probe
                self.authentic_impostor[i, 0 : min(i + 1, len(self.gallery))] = -1

        self.matches = None

    def get_features_label(self, feature_path):
        subject_id = path.split(feature_path)[1]
        feature_label = path.join(
            path.split(path.split(feature_path)[0])[1], subject_id[:-4]
        )

        if self.dataset_name == "CHIYA":
            subject_id = subject_id[:-5]

        elif self.dataset_name == "CHIYA_VAL":
            subject_id = feature_label[1:-4]

        elif self.dataset_name == "PUBLIC_IVS":
            subject_id = path.split(feature_label)[0]

        elif self.id_length > 0:
            subject_id = subject_id[: self.id_length]
        else:
            subject_id = subject_id.split("_")[0]

        return subject_id, feature_label

    def get_features(self, file):
        all_features = []
        all_labels = []
        all_subject_ids = []

        for j in range(len(file)):
            image_path = file[j]
            features = np.load(image_path)
            subject_id, feature_label = self.get_features_label(image_path)

            all_features.append(features)
            all_subject_ids.append(subject_id)
            all_labels.append(feature_label)

        return (
            np.asarray(all_features),
            np.asarray(all_subject_ids),
            np.asarray(all_labels),
        )

    def match_features(self):
        self.matches = cosine_similarity(self.probe, self.gallery)

    def create_label_indices(self, labels):
        indices = np.linspace(0, len(labels) - 1, len(labels)).astype(int)
        return np.transpose(np.vstack([indices, labels]))

    def get_indices_score(self, auth_or_imp):
        x, y = np.where(self.authentic_impostor == auth_or_imp)
        return np.transpose(
            np.vstack(
                [
                    x,
                    y,
                    np.round(self.matches[self.authentic_impostor == auth_or_imp], 6),
                ]
            )
        )

    def save_matches(self, output, group):
        np.save(path.join(output, f"{group}_authentic.npy"), self.get_indices_score(1))
        np.save(path.join(output, f"{group}_impostor.npy"), self.get_indices_score(0))
        np.savetxt(
            path.join(output, f"{group}_labels.txt"),
            self.create_label_indices(self.probe_labels),
            delimiter=" ",
            fmt="%s",
        )
        if not self.probe_equal_gallery:
            np.savetxt(
                path.join(output, f"{group}_gallery_labels.txt"),
                self.create_label_indices(self.gallery_labels),
                delimiter=" ",
                fmt="%s",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match Extracted Features")
    parser.add_argument("-probe", "-p", help="Probe image list.")
    parser.add_argument("-gallery", "-g", help="Gallery image list.")
    parser.add_argument("-output", "-o", help="Output folder.")
    parser.add_argument("-dataset", "-d", help="Dataset name.")
    parser.add_argument("-group", "-gr", help="Group name, e.g. AA")

    args = parser.parse_args()
    time1 = datetime.now()

    if not path.exists(args.output):
        makedirs(args.output)

    matcher = Matcher(args.probe, args.gallery, args.dataset.upper())
    matcher.match_features()
    matcher.save_matches(args.output, args.group)

    time2 = datetime.now()
    print(f"Total time to match: {time2 - time1}")
