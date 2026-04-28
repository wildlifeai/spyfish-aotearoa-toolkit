import io
import zipfile
from unittest.mock import Mock, patch

import pandas as pd


@patch("sftk.external.biigle_api.Api")
def test_get_projects(mock_api_class):
    """BiigleHandler should retrieve list of projects."""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"id": 1, "name": "Project 1"},
        {"id": 2, "name": "Project 2"},
    ]

    mock_api_instance = Mock()
    mock_api_instance.get.return_value = mock_response
    mock_api_class.return_value = mock_api_instance

    from sftk.biigle_handler import BiigleHandler

    handler = BiigleHandler("test@example.com", "test_token")
    projects = handler.get_projects()

    mock_api_instance.get.assert_called_once_with("projects")
    assert len(projects) == 2
    assert projects[0]["name"] == "Project 1"


def test_get_volumes():
    """BiigleHandler should retrieve list of volumes from a project."""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"id": 1, "name": "Volume 1"},
        {"id": 2, "name": "Volume 2"},
    ]

    mock_api_instance = Mock()
    mock_api_instance.get.return_value = mock_response

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        volumes = handler.get_volumes(project_id=3711)

        mock_api_instance.get.assert_called_once_with("projects/3711/volumes")
        assert len(volumes) == 2


def test_create_pending_volume():
    """BiigleHandler should create pending volumes with default and custom media types."""
    mock_response = Mock()
    mock_response.json.return_value = {"id": 12345, "name": None, "url": None}

    mock_api_instance = Mock()
    mock_api_instance.post.return_value = mock_response

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")

        # Test default media type (video)
        pending_volume = handler.create_pending_volume(project_id=3711)
        mock_api_instance.post.assert_called_with(
            "projects/3711/pending-volumes", json={"media_type": "video"}
        )
        assert pending_volume["id"] == 12345

        # Test custom media type
        pending_volume = handler.create_pending_volume(
            project_id=3711, media_type="image"
        )
        mock_api_instance.post.assert_called_with(
            "projects/3711/pending-volumes", json={"media_type": "image"}
        )


def test_setup_volume_with_files():
    """BiigleHandler should configure pending volume with files."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "id": 12345,
        "name": "test_volume",
        "url": "s3://test",
    }

    mock_api_instance = Mock()
    mock_api_instance.put.return_value = mock_response

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        result = handler.setup_volume_with_files(
            12345,
            "test_volume",
            "disk-98://biigle_clips/test/",
            ["file1.mp4", "file2.mp4"],
        )

        expected_payload = {
            "name": "test_volume",
            "url": "disk-98://biigle_clips/test/",
            "files": ["file1.mp4", "file2.mp4"],
        }
        mock_api_instance.put.assert_called_once_with(
            "pending-volumes/12345", json=expected_payload
        )
        assert result["name"] == "test_volume"


def test_create_volume_from_s3_files():
    """BiigleHandler should create and configure volume from S3 files."""
    mock_pending_response = Mock()
    mock_pending_response.json.return_value = {"id": 12345}

    mock_setup_response = Mock()
    mock_setup_response.json.return_value = {"id": 12345, "name": "test_volume"}

    mock_api_instance = Mock()
    mock_api_instance.post.return_value = mock_pending_response
    mock_api_instance.put.return_value = mock_setup_response

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        result = handler.create_volume_from_s3_files(
            project_id=3711,
            volume_name="test_volume",
            s3_url="disk-98://biigle_clips/test/",
            files=["file1.mp4", "file2.mp4"],
        )

        assert result["id"] == 12345
        assert mock_api_instance.post.call_count == 1
        assert mock_api_instance.put.call_count == 1


def test_create_label_tree():
    """BiigleHandler should create label tree and add labels."""
    mock_tree_response = Mock()
    mock_tree_response.json.return_value = {"id": 456, "name": "Test Tree"}

    mock_label_response = Mock()
    mock_label_response.json.return_value = {"id": 789, "name": "Test Species"}

    mock_api_instance = Mock()
    mock_api_instance.post.side_effect = [
        mock_tree_response,
        mock_label_response,
        mock_label_response,
    ]

    labels_df = pd.DataFrame(
        {
            "name": ["Species 1", "Species 2"],
            "color": ["FF0000", "00FF00"],
            "source_id": [123, 456],
        }
    )

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        with patch("pandas.read_csv", return_value=labels_df):
            from sftk.biigle_handler import BiigleHandler

            handler = BiigleHandler("test@example.com", "test_token")
            result = handler.create_label_tree(
                csv_path="test_labels.csv",
                tree_name="Test Tree",
                tree_description="Test Description",
                project_id=3711,
            )

            assert mock_api_instance.post.call_count == 3  # 1 tree + 2 labels
            assert result["tree_id"] == 456
            assert len(result["labels"]) == 2


def test_create_report():
    """BiigleHandler should create report and return report_id."""
    mock_response = Mock()
    mock_response.json.return_value = {"id": 999}
    mock_response.raise_for_status = Mock()

    mock_api_instance = Mock()
    mock_api_instance.post.return_value = mock_response

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        report_id = handler.create_report("volumes", resource_id=12345, type_id=8)

        mock_api_instance.post.assert_called_once_with(
            "volumes/12345/reports", json={"type_id": 8}
        )
        assert report_id == 999


def test_download_report_zip_bytes():
    """BiigleHandler should download report ZIP with polling support."""
    # Test immediate download
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"fake zip content"

    mock_api_instance = Mock()
    mock_api_instance.get.return_value = mock_response

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        zip_bytes = handler.download_report_zip_bytes(report_id=999)

        mock_api_instance.get.assert_called_once_with(
            "reports/999", raise_for_status=False
        )
        assert zip_bytes == b"fake zip content"

    # Test polling
    mock_not_ready = Mock()
    mock_not_ready.status_code = 404
    mock_ready = Mock()
    mock_ready.status_code = 200
    mock_ready.content = b"fake zip content"

    mock_api_instance2 = Mock()
    mock_api_instance2.get.side_effect = [mock_not_ready, mock_ready]

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance2):
        with patch("time.sleep"):
            handler = BiigleHandler("test@example.com", "test_token")
            zip_bytes = handler.download_report_zip_bytes(
                report_id=999, max_tries=5, poll_interval=0.1
            )
            assert mock_api_instance2.get.call_count == 2


def test_read_csvs_from_zip_bytes():
    """BiigleHandler should read CSVs from ZIP bytes including nested ZIPs."""
    # Test simple ZIP
    csv1_content = "col1,col2\n1,2\n3,4"
    csv2_content = "col1,col2\n5,6\n7,8"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("file1.csv", csv1_content)
        zf.writestr("file2.csv", csv2_content)
    zip_bytes = zip_buffer.getvalue()

    with patch("sftk.biigle_handler.Api", return_value=Mock()):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        csv_dict = handler.read_csvs_from_zip_bytes(zip_bytes)

        assert len(csv_dict) == 2
        assert "file1.csv" in csv_dict
        assert len(csv_dict["file1.csv"]) == 2

    # Test nested ZIP
    inner_csv = "col1,col2\n1,2"
    inner_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_zip_buffer, "w") as zf:
        zf.writestr("inner_file.csv", inner_csv)

    outer_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_zip_buffer, "w") as zf:
        zf.writestr("inner.zip", inner_zip_buffer.getvalue())

    with patch("sftk.biigle_handler.Api", return_value=Mock()):
        handler = BiigleHandler("test@example.com", "test_token")
        csv_dict = handler.read_csvs_from_zip_bytes(
            outer_zip_buffer.getvalue(), allow_nested=True
        )
        assert "inner_file.csv" in csv_dict


def test_concat_csv_dict():
    """BiigleHandler should concatenate CSV dict with source column."""
    with patch("sftk.biigle_handler.Api", return_value=Mock()):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        csv_dict = {
            "file1.csv": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
            "file2.csv": pd.DataFrame({"col1": [5, 6], "col2": [7, 8]}),
        }

        result = handler.concat_csv_dict(csv_dict, source_col="source_file")

        assert len(result) == 4
        assert "source_file" in result.columns
        assert set(result["source_file"].unique()) == {"file1.csv", "file2.csv"}


def test_export_report_to_df():
    """BiigleHandler should create report, download ZIP, and return DataFrame."""
    mock_report_response = Mock()
    mock_report_response.json.return_value = {"id": 999}
    mock_report_response.raise_for_status = Mock()

    csv_content = "col1,col2\n1,2\n3,4"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("annotations.csv", csv_content)

    mock_download_response = Mock()
    mock_download_response.status_code = 200
    mock_download_response.content = zip_buffer.getvalue()

    mock_api_instance = Mock()
    mock_api_instance.post.return_value = mock_report_response
    mock_api_instance.get.return_value = mock_download_response

    with patch("sftk.biigle_handler.Api", return_value=mock_api_instance):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")
        result_df = handler.export_report_to_df(
            resource="volumes", resource_id=12345, type_id=8
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "source_file" in result_df.columns


def test_build_s3_url():
    """BiigleHandler should build correct BIIGLE S3 URL format."""
    with patch("sftk.biigle_handler.Api", return_value=Mock()):
        from sftk.biigle_handler import BiigleHandler

        handler = BiigleHandler("test@example.com", "test_token")

        # Test default disk_id and trailing slash handling
        assert (
            handler.build_s3_url("biigle_clips/test_folder")
            == "disk-134://biigle_clips/test_folder/"
        )
        assert (
            handler.build_s3_url("biigle_clips/test_folder/")
            == "disk-134://biigle_clips/test_folder/"
        )

        # Test custom disk_id
        assert (
            handler.build_s3_url("biigle_clips/test", disk_id=99)
            == "disk-99://biigle_clips/test/"
        )
