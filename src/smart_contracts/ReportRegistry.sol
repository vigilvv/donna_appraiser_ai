// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReportRegistry {
    event ReportStored(string uuid, string hash, uint timestamp);

    struct Report {
        string hash;
        uint timestamp;
    }

    // uuid => list of reports
    mapping(string => Report[]) public reports;

    function storeHash(string memory uuid, string memory hash) public {
        reports[uuid].push(Report(hash, block.timestamp));
        emit ReportStored(uuid, hash, block.timestamp);
    }

    function getReports(
        string memory uuid
    ) public view returns (Report[] memory) {
        return reports[uuid];
    }
}
