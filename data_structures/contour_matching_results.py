from skimage.transform import EuclideanTransform
import numpy.typing as npt
import utils.assertions as assertions


class ContourMatchingSingleStepResults:
    def __init__(
        self,
        transform: EuclideanTransform,
        source_matching_indices: npt.NDArray[int],
        target_matching_indices: npt.NDArray[int],
    ) -> None:
        assertions.assert_2d_rigid_transform(transform)
        assertions.assert_flat_array(source_matching_indices, int)
        assertions.assert_flat_array(target_matching_indices, int)
        self.transform: EuclideanTransform = transform
        self.source_matching_indices: npt.NDArray[int] = source_matching_indices
        self.target_matching_indices: npt.NDArray[int] = target_matching_indices

    def inverse(self) -> "ContourMatchingSingleStepResults":
        return ContourMatchingSingleStepResults(
            self.transform.inverse,
            self.target_matching_indices,
            self.source_matching_indices,
        )


class ContourMatchingResults:
    def __init__(
        self,
        init_matching_results: ContourMatchingSingleStepResults,
        icp_refined_transform: EuclideanTransform,
        refined_matching_result: ContourMatchingSingleStepResults,
        valid: bool,
        source_contour_index: int,
        target_contour_index: int,
    ) -> None:
        assertions.assert_2d_rigid_transform(icp_refined_transform)
        assert source_contour_index >= 0
        assert target_contour_index >= 0
        self.init_matching_results: ContourMatchingSingleStepResults = (
            init_matching_results
        )
        self.icp_refined_transform: EuclideanTransform = icp_refined_transform
        self.refined_matching_result: ContourMatchingSingleStepResults = (
            refined_matching_result
        )
        self.valid: bool = valid
        self.source_contour_index: int = source_contour_index
        self.target_contour_index: int = target_contour_index

    def inverse(self) -> "ContourMatchingResults":
        return ContourMatchingResults(
            self.init_matching_results.inverse(),
            self.icp_refined_transform.inverse(),
            self.refined_matching_result.inverse(),
            self.valid,
            self.target_contour_index,
            self.source_contour_index,
        )
